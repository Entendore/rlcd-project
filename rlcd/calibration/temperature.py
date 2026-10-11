"""Input-adaptive temperature scaling, fitted to minimize ECE."""
from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F

from rlcd.core.reward import TaskComplexityEstimator

DOMAIN_FACTORS = {"medical": 0.80, "financial": 0.85, "general": 1.00}


class AdaptiveTemperatureCalibrator:
    def __init__(self, base_temperature: float = 1.0,
                 search_range: Tuple[float, float] = (0.5, 2.5), bins: int = 15):
        self.base_T = base_temperature
        self.search_range = search_range
        self.bins = bins
        self.complexity = TaskComplexityEstimator()
        self._fitted_T = None

    def dynamic_temperature(self, state_text: str, domain: str = "general") -> float:
        c = self.complexity.estimate(state_text, num_options=8)
        domain_factor = DOMAIN_FACTORS.get(domain, 1.0)
        base = self._fitted_T if self._fitted_T is not None else self.base_T
        return float(np.clip(base * (1 + 0.5 * c) * domain_factor, *self.search_range))

    def expected_calibration_error(self, probs: np.ndarray, outcomes: np.ndarray) -> float:
        conf = probs.max(axis=-1)
        pred = probs.argmax(axis=-1)
        correct = (pred == np.asarray(outcomes)).astype(float)
        ece, n = 0.0, len(conf)
        edges = np.linspace(0, 1, self.bins + 1)
        for i in range(self.bins):
            m = (conf > edges[i]) & (conf <= edges[i + 1])
            if m.sum() == 0:
                continue
            ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
        return float(ece)

    def fit(self, all_logits, all_outcomes, complexities, domains) -> float:
        """Grid-search the global temperature minimizing ECE on held-out data."""
        if torch.is_tensor(all_outcomes):
            outcomes = all_outcomes.detach().cpu().numpy()
        else:
            outcomes = np.asarray(all_outcomes)
        logits = all_logits.detach().cpu()
        best_T, best_ece = self.base_T, float("inf")
        for T in np.linspace(*self.search_range, 60):
            probs = F.softmax(logits / float(T), dim=-1).numpy()
            ece = self.expected_calibration_error(probs, outcomes)
            if ece < best_ece:
                best_ece, best_T = ece, float(T)
        self._fitted_T = best_T
        return best_T

    def calibrate_choice(self, logits, state_text, domain="general"):
        T = self.dynamic_temperature(state_text, domain)
        return F.softmax(logits / T, dim=-1)

    def calibrate_noul(self, logit, state_text, domain="general"):
        T = self.dynamic_temperature(state_text, domain)
        return torch.sigmoid(logit / T)
