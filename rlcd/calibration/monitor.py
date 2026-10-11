"""Production calibration monitoring: rolling metrics, drift detection with
tiered alerts (soft/medium/hard), LRU cache, automated recalibration."""
from __future__ import annotations
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


class CalibrationCache:
    """LRU cache keyed on (state_hash, question_hash)."""
    def __init__(self, max_size: int = 50_000):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.hits = self.misses = 0

    def get(self, state_hash: str, question_hash: str):
        key = (state_hash, question_hash)
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, state_hash: str, question_hash: str, value: dict):
        key = (state_hash, question_hash)
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[key] = {**value, "cached_at": time.time()}

    def stats(self) -> Dict[str, float]:
        total = self.hits + self.misses
        return {"hit_rate": self.hits / max(total, 1), "size": len(self.cache)}


@dataclass
class CalibrationStats:
    ece: float = 0.0
    accuracy: float = 0.0
    brier: float = 0.0
    n: int = 0
    timestamp: float = field(default_factory=time.time)


class CalibrationMonitor:
    SOFT, MEDIUM, HARD = "soft", "medium", "hard"

    def __init__(self, window=1000, ece_threshold=0.02, acc_threshold=0.03,
                 soft=0.04, medium=0.08, bins=15):
        self.window, self.ece_threshold, self.acc_threshold = window, ece_threshold, acc_threshold
        self.soft_t, self.medium_t = soft, medium
        self.bins = bins
        self.confidences: List[float] = []
        self.corrects: List[float] = []
        self.baseline: Optional[CalibrationStats] = None
        self.alerts: List[Dict] = []

    def update(self, confidences: List[float], corrects: List[float]) -> CalibrationStats:
        self.confidences.extend(confidences)
        self.corrects.extend(corrects)
        self.confidences = self.confidences[-self.window:]
        self.corrects = self.corrects[-self.window:]
        stats = self._compute_stats()
        if self.baseline is None and len(self.corrects) >= self.window:
            self.baseline = stats
        elif self.baseline is not None:
            self._check_drift(stats)
        return stats

    def _compute_stats(self) -> CalibrationStats:
        conf = np.array(self.confidences)
        cor = np.array(self.corrects)
        n = len(cor)
        if n == 0:
            return CalibrationStats()
        ece = 0.0
        edges = np.linspace(0, 1, self.bins + 1)
        for i in range(self.bins):
            m = (conf > edges[i]) & (conf <= edges[i + 1])
            if m.sum():
                ece += (m.sum() / n) * abs(cor[m].mean() - conf[m].mean())
        return CalibrationStats(ece=float(ece), accuracy=float(cor.mean()),
                                brier=float(np.mean((conf - cor) ** 2)), n=n)

    def _check_drift(self, current: CalibrationStats) -> Optional[Dict]:
        ece_drift = abs(current.ece - self.baseline.ece)
        acc_drift = abs(current.accuracy - self.baseline.accuracy)
        if ece_drift > self.medium_t:
            severity, action = self.HARD, "full_retrain"
        elif ece_drift > self.soft_t:
            severity, action = self.MEDIUM, "refit_calibration_layer"
        elif ece_drift > self.ece_threshold:
            severity, action = self.SOFT, "adjust_temperature"
        else:
            return None
        alert = {"severity": severity, "action": action,
                 "ece_drift": round(ece_drift, 4), "acc_drift": round(acc_drift, 4),
                 "current_ece": round(current.ece, 4),
                 "baseline_ece": round(self.baseline.ece, 4),
                 "timestamp": time.time()}
        self.alerts.append(alert)
        return alert

    def reliability_diagram(self) -> List[Dict]:
        conf, cor = np.array(self.confidences), np.array(self.corrects)
        out, edges = [], np.linspace(0, 1, self.bins + 1)
        for i in range(self.bins):
            m = (conf > edges[i]) & (conf <= edges[i + 1])
            if m.sum():
                out.append({"bin_center": (edges[i] + edges[i + 1]) / 2,
                            "confidence": float(conf[m].mean()),
                            "accuracy": float(cor[m].mean()),
                            "count": int(m.sum())})
        return out


class AutomatedRecalibrator:
    """Collects production (prediction, outcome) feedback; refits the
    calibration layer on fresh data when the buffer fills."""
    def __init__(self, calibrator, buffer_size: int = 5000):
        self.calibrator = calibrator
        self.buffer_size = buffer_size
        self.buffer: List[Tuple[np.ndarray, np.ndarray]] = []

    def add_feedback(self, probs: np.ndarray, outcomes: np.ndarray):
        self.buffer.append((probs, outcomes))
        if len(self.buffer) >= self.buffer_size:
            self.recalibrate()

    def recalibrate(self) -> Optional[float]:
        if not self.buffer:
            return None
        probs = np.concatenate([np.atleast_1d(p) for p, _ in self.buffer])
        outs = np.concatenate([np.atleast_1d(o) for _, o in self.buffer])
        if probs.ndim == 1:
            probs = np.stack([probs, 1 - probs], axis=-1)
        logits = torch.tensor(np.log(probs + 1e-9), dtype=torch.float32)
        outs_t = torch.tensor(outs, dtype=torch.long)
        T = self.calibrator.fit(logits, outs_t,
                                complexities=np.zeros(len(outs)),
                                domains=["general"] * len(outs))
        self.buffer.clear()
        return T
