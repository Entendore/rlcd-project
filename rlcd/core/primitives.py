"""TypeSafe primitives: Choice, Score, Noul — typed questions & calibrated answers."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np


class PrimitiveType(str, Enum):
    CHOICE = "choice"
    SCORE = "score"
    NOUL = "noul"


@dataclass
class Question:
    id: str
    type: PrimitiveType
    instructions: str
    criteria: Optional[Dict[str, str]] = None
    scale: Optional[List[Any]] = None
    weight: float = 1.0

    def __post_init__(self):
        if self.type == PrimitiveType.CHOICE and not self.criteria:
            raise ValueError(f"Choice question '{self.id}' requires criteria/options.")
        if self.type == PrimitiveType.SCORE and not self.scale:
            raise ValueError(f"Score question '{self.id}' requires a scale.")
        if len(self.criteria or {}) > 255:
            raise ValueError("Choice supports up to 255 options.")

    def validate_answer_space(self, output: Any) -> bool:
        if self.type == PrimitiveType.CHOICE:
            return output in self.criteria
        if self.type == PrimitiveType.SCORE:
            return self.scale[0] <= output <= self.scale[-1]
        if self.type == PrimitiveType.NOUL:
            return 0.0 <= output <= 1.0
        return False

    def hash_key(self) -> str:
        payload = json.dumps({
            "id": self.id, "type": self.type.value,
            "instructions": self.instructions,
            "criteria": self.criteria, "scale": self.scale,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class UncertaintyBreakdown:
    aleatoric: float = 0.0
    epistemic: float = 0.0
    distributional: float = 0.0

    @property
    def total(self) -> float:
        return float(np.clip(self.aleatoric + self.epistemic + self.distributional, 0, 1))

    def to_dict(self) -> Dict[str, float]:
        return {"aleatoric": round(self.aleatoric, 4),
                "epistemic": round(self.epistemic, 4),
                "distributional": round(self.distributional, 4),
                "total": round(self.total, 4)}


@dataclass
class Answer:
    question_id: str
    type: PrimitiveType
    value: Any
    probabilities: Optional[Dict[str, float]] = None
    confidence: float = 0.0
    uncertainty: UncertaintyBreakdown = field(default_factory=UncertaintyBreakdown)
    explanation: Optional[str] = None
    audit_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {"question_id": self.question_id, "type": self.type.value,
               "value": self.value, "confidence": round(self.confidence, 4),
               "uncertainty": self.uncertainty.to_dict()}
        if self.probabilities:
            out["probabilities"] = {k: round(v, 4) for k, v in self.probabilities.items()}
        if self.explanation:
            out["explanation"] = self.explanation
        if self.audit_id:
            out["audit_id"] = self.audit_id
        return out


def noul_confidence_from_probability(p_true: float) -> float:
    return abs(p_true - 0.5) * 2.0


def choice_confidence_from_distribution(probs: Dict[str, float]) -> float:
    if not probs:
        return 0.0
    s = sorted(probs.values(), reverse=True)
    top, second = s[0], (s[1] if len(s) > 1 else 0.0)
    return float(np.clip(top - second, 0.0, 1.0))


def score_confidence_from_distribution(dist: List[float]) -> float:
    d = np.asarray(dist, dtype=np.float64)
    d = d / (d.sum() + 1e-12)
    entropy = -(d * np.log(d + 1e-12)).sum()
    return float(1.0 - entropy / (np.log(len(d) + 1e-12)))
