"""Adaptive RLCD rewards: R = a*Accuracy + b*Calibration - g*Inconsistency,
with a,b,g adjusted DYNAMICALLY per domain risk and task complexity,
plus the log-score ('Rewarding Doubt') proper scoring rule."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import torch


class TaskComplexityEstimator:
    HEURISTIC_WEIGHTS = {"length": 0.3, "ambiguity": 0.4, "rarity": 0.3}

    def estimate(self, state_text: str, num_options: int) -> float:
        length_c = min(len(state_text) / 4000.0, 1.0)
        words = state_text.split()
        uniq = len(set(words)) / max(len(words), 1)
        ambiguity_c = 1.0 - uniq
        choice_c = min(num_options / 64.0, 1.0)
        w = self.HEURISTIC_WEIGHTS
        return float(np.clip(w["length"] * length_c + w["ambiguity"] * ambiguity_c
                             + w["rarity"] * choice_c, 0.0, 1.0))


@dataclass(frozen=True)
class DomainRiskProfile:
    name: str
    calibration_weight: float
    accuracy_weight: float
    consistency_weight: float


DOMAIN_PROFILES = {
    "financial": DomainRiskProfile("financial", 0.60, 0.30, 0.10),
    "medical":   DomainRiskProfile("medical",   0.70, 0.20, 0.10),
    "general":   DomainRiskProfile("general",   0.50, 0.40, 0.10),
}


class AdaptiveRewardFunction:
    def __init__(self, base_alpha=0.5, base_beta=0.35, base_gamma=0.15,
                 domain_profile: Optional[DomainRiskProfile] = None):
        self.base_alpha, self.base_beta, self.base_gamma = base_alpha, base_beta, base_gamma
        self.domain = domain_profile or DOMAIN_PROFILES["general"]
        self.complexity_estimator = TaskComplexityEstimator()

    def get_weights(self, state_text: str, num_options: int) -> Dict[str, float]:
        c = self.complexity_estimator.estimate(state_text, num_options)
        alpha = self.base_alpha * (1 - c) * (self.domain.accuracy_weight / 0.40)
        beta = self.base_beta * (1 + c) * (self.domain.calibration_weight / 0.50)
        gamma = self.base_gamma * (1 - c)
        total = alpha + beta + gamma
        return {"alpha": alpha / total, "beta": beta / total,
                "gamma": gamma / total, "complexity": c}

    @staticmethod
    def accuracy_reward(pred_correct: torch.Tensor) -> torch.Tensor:
        return pred_correct.float()

    @staticmethod
    def calibration_reward(pred_prob: torch.Tensor, outcome: torch.Tensor) -> torch.Tensor:
        return -((pred_prob - outcome.float()) ** 2)

    @staticmethod
    def consistency_penalty(pred_prob: torch.Tensor,
                            neighbor_probs: torch.Tensor) -> torch.Tensor:
        if neighbor_probs.numel() == 0:
            return torch.zeros_like(pred_prob)
        return (pred_prob.unsqueeze(1) - neighbor_probs).abs().mean(dim=1)

    def compute(self, pred_correct, pred_prob, outcome, neighbor_probs,
                state_text: str, num_options: int) -> Dict[str, torch.Tensor]:
        w = self.get_weights(state_text, num_options)
        acc = self.accuracy_reward(pred_correct)
        cal = self.calibration_reward(pred_prob, outcome)
        inc = (self.consistency_penalty(pred_prob, neighbor_probs)
               if neighbor_probs is not None else torch.zeros_like(acc))
        total = w["alpha"] * acc + w["beta"] * cal - w["gamma"] * inc
        return {"reward": total, "accuracy": acc, "calibration": cal,
                "inconsistency": inc, "weights": w}

    @staticmethod
    def log_score_reward(pred_prob, outcome, eps=1e-9):
        """Log scoring rule: optimal policy = perfectly calibrated confidence."""
        p = pred_prob.clamp(eps, 1 - eps)
        return outcome.float() * torch.log(p) + (1 - outcome.float()) * torch.log(1 - p)
