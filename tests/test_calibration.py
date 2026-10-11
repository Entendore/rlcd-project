import numpy as np
import torch
import pytest

from rlcd.calibration.temperature import AdaptiveTemperatureCalibrator
from rlcd.calibration.monitor import CalibrationCache, CalibrationMonitor
from rlcd.evaluation.metrics import expected_calibration_error


def test_temperature_fitting_reduces_ece():
    torch.manual_seed(0)
    N, C = 4000, 4
    logits = torch.randn(N, C)
    labels = torch.randint(0, C, (N,))
    cal = AdaptiveTemperatureCalibrator(search_range=(0.5, 3.0))
    T = cal.fit(logits, labels, complexities=np.zeros(N), domains=["general"] * N)
    assert 0.5 <= T <= 3.0
    probs_fit = torch.softmax(logits / T, dim=-1).numpy()
    probs_raw = torch.softmax(logits, dim=-1).numpy()
    ece_fit = cal.expected_calibration_error(probs_fit, labels.numpy())
    ece_raw = cal.expected_calibration_error(probs_raw, labels.numpy())
    assert ece_fit <= ece_raw + 1e-6


def test_cache_hit_miss_eviction():
    c = CalibrationCache(max_size=2)
    c.put("s", "q1", {"answer": {"value": 1}})
    assert c.get("s", "q1") is not None
    assert c.get("s", "zz") is None
    c.put("s", "q2", {"answer": {"value": 2}})
    c.put("s", "q3", {"answer": {"value": 3}})   # evicts q1 (LRU)
    assert c.get("s", "q1") is None
    assert c.stats()["hit_rate"] > 0


def test_monitor_drift_alert():
    m = CalibrationMonitor(window=100, ece_threshold=0.01, soft=0.02, medium=0.05)
    rs0, rs1 = np.random.RandomState(0), np.random.RandomState(1)
    confs = np.full(100, 0.8)
    corr = (rs0.random(100) < 0.8).astype(float)
    m.update(confs.tolist(), corr.tolist())
    confs2 = np.full(100, 0.95)
    corr2 = (rs1.random(100) < 0.5).astype(float)
    m.update(confs2.tolist(), corr2.tolist())
    assert len(m.alerts) >= 1
    assert m.alerts[-1]["severity"] in {"soft", "medium", "hard"}


def test_reliability_diagram_shape():
    m = CalibrationMonitor(window=50)
    m.update(np.linspace(0.05, 0.95, 50).tolist(),
             (np.linspace(0.05, 0.95, 50) > 0.5).astype(float).tolist())
    diag = m.reliability_diagram()
    assert len(diag) > 0 and {"confidence", "accuracy", "count"} <= set(diag[0])
