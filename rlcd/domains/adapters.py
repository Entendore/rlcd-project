"""Domain-specific adapters: healthcare (zero-tolerance safety) and
financial (regulatory compliance + audit trails)."""
from __future__ import annotations
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict

from rlcd.core.reward import DOMAIN_PROFILES
from rlcd.core.primitives import Answer


class BaseDomainAdapter(ABC):
    name = "general"
    profile = DOMAIN_PROFILES["general"]
    min_confidence_for_action: float = 0.80
    max_uncertainty_tolerance: float = 0.15

    @abstractmethod
    def apply(self, answer: Answer, context: Dict[str, Any]) -> Answer: ...


class GeneralAdapter(BaseDomainAdapter):
    def apply(self, answer, context):
        return answer


class HealthcareAdapter(BaseDomainAdapter):
    name = "medical"
    profile = DOMAIN_PROFILES["medical"]
    min_confidence_for_action = 0.95
    max_uncertainty_tolerance = 0.02

    def apply(self, answer: Answer, context: Dict[str, Any]) -> Answer:
        high_stakes = context.get("high_stakes", True)
        if answer.confidence < self.min_confidence_for_action:
            answer.value = "require_human_review"
            answer.explanation = (f"Confidence {answer.confidence:.3f} below medical "
                                  f"safety threshold {self.min_confidence_for_action}")
            answer.confidence = 0.0
            return answer
        if answer.uncertainty.total > self.max_uncertainty_tolerance:
            answer.value = "require_additional_testing"
            answer.explanation = (f"Uncertainty {answer.uncertainty.total:.3f} exceeds "
                                  f"tolerance {self.max_uncertainty_tolerance}")
            return answer
        if high_stakes:
            answer.explanation = (f"HIGH-STAKES: recommendation '{answer.value}' with "
                                  f"confidence {answer.confidence:.3f} — physician "
                                  f"sign-off required.")
        return answer


class FinancialAdapter(BaseDomainAdapter):
    name = "financial"
    profile = DOMAIN_PROFILES["financial"]
    min_confidence_for_action = 0.90
    max_uncertainty_tolerance = 0.05

    def apply(self, answer: Answer, context: Dict[str, Any]) -> Answer:
        answer.confidence = round(answer.confidence * 0.90, 4)   # conservative bias
        answer.audit_id = str(uuid.uuid4())
        answer.explanation = (f"model=rlcd-v1; domain=financial; decision={answer.value}; "
                              f"conf={answer.confidence}; timestamp="
                              f"{datetime.now(timezone.utc).isoformat()}")
        if answer.confidence < self.min_confidence_for_action:
            answer.value = "escalate_compliance_team"
        return answer


DOMAIN_REGISTRY = {
    "general": GeneralAdapter(),
    "medical": HealthcareAdapter(),
    "financial": FinancialAdapter(),
}
