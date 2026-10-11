"""Calibration evaluation: ECE, MCE, Brier, NLL, reliability diagrams."""
from __future__ import annotations
from typing import Dict, List
import numpy as np


def expected_calibration_error(confidences, corrects, bins=15) -> float:
    conf, cor = np.asarray(confidences), np.asarray(corrects)
    ece, n = 0.0, len(conf)
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            ece += (m.sum() / n) * abs(cor[m].mean() - conf[m].mean())
    return float(ece)


def maximum_calibration_error(confidences, corrects, bins=15) -> float:
    conf, cor = np.asarray(confidences), np.asarray(corrects)
    worst, edges = 0.0, np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            worst = max(worst, abs(cor[m].mean() - conf[m].mean()))
    return float(worst)


def brier_score(confidences, corrects) -> float:
    return float(np.mean((np.asarray(confidences) - np.asarray(corrects)) ** 2))


def negative_log_likelihood(confidences, corrects, eps=1e-9) -> float:
    p = np.clip(np.asarray(confidences), eps, 1 - eps)
    y = np.asarray(corrects)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_diagram_data(confidences, corrects, bins=15) -> List[Dict]:
    conf, cor = np.asarray(confidences), np.asarray(corrects)
    out, edges = [], np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            out.append({"confidence": float(conf[m].mean()),
                        "accuracy": float(cor[m].mean()),
                        "count": int(m.sum()),
                        "gap": float(abs(cor[m].mean() - conf[m].mean()))})
    return out


def full_report(confidences, corrects) -> Dict[str, float]:
    return {"ece": round(expected_calibration_error(confidences, corrects), 4),
            "mce": round(maximum_calibration_error(confidences, corrects), 4),
            "brier": round(brier_score(confidences, corrects), 4),
            "nll": round(negative_log_likelihood(confidences, corrects), 4),
            "accuracy": round(float(np.mean(corrects)), 4),
            "mean_confidence": round(float(np.mean(confidences)), 4)}
