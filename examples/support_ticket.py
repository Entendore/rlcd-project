"""End-to-end demo: enhanced support-ticket routing with adaptive
calibration, uncertainty decomposition, hybrid paths, confidence routing."""
import numpy as np
import torch

from rlcd.core.model import RLCDDecisionModel
from rlcd.core.reward import AdaptiveRewardFunction, DOMAIN_PROFILES
from rlcd.calibration.temperature import AdaptiveTemperatureCalibrator
from rlcd.calibration.uncertainty import UncertaintyDecomposer
from rlcd.calibration.monitor import CalibrationCache
from rlcd.inference.engine import RLCDInferenceEngine, RoutingThresholds
from rlcd.core.primitives import Question, PrimitiveType
from rlcd.evaluation.metrics import full_report


def build_model():
    from types import SimpleNamespace
    cfg = SimpleNamespace(hidden_dim=256, num_layers=4, num_heads=4,
                          vocab_size=1000, max_state_length=256,
                          num_experts=4, num_experts_per_token=2, dropout=0.1)
    return RLCDDecisionModel(cfg)


def main():
    torch.manual_seed(42)
    model = build_model()
    engine = RLCDInferenceEngine(
        model=model,
        calibrator=AdaptiveTemperatureCalibrator(),
        uncertainty=UncertaintyDecomposer(model, mc_samples=5),
        cache=CalibrationCache(max_size=10_000),
        thresholds=RoutingThresholds(),
        fast_path_conf=0.90, deep_path_conf=0.60,
        quantize_mode="none", domain="general")

    state = ("Hi, I've been trying to connect my Stripe account for "
             "3 days and it keeps failing. I'm losing sales. Please help ASAP.")
    questions = [
        Question(id="department", type=PrimitiveType.CHOICE,
                 instructions="Which team should handle this message?",
                 criteria={"billing": "Charges, invoices, refunds",
                           "technical": "Bugs, outages, integration",
                           "account": "Login/access issues",
                           "other": "None of the above"}),
        Question(id="is_urgent", type=PrimitiveType.NOUL,
                 instructions="Does the message convey urgency?"),
        Question(id="frustration", type=PrimitiveType.SCORE,
                 instructions="How frustrated is this customer?",
                 scale=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ]
    tokens = torch.randint(1, 1000, (1, 64))
    result = engine.answer(state, questions, tokens=tokens, num_options=4)

    print("=== Support Ticket Decision ===")
    for qid, ans in result["answers"].items():
        print(f"\n[{qid}]  value={ans['value']}  confidence={ans['confidence']}")
        print(f"   uncertainty={ans['uncertainty']}")
        print(f"   {ans['explanation']}")
    print(f"\nLatency: {result['latency_ms']} ms | "
          f"cache hit-rate: {result['cache_stats']['hit_rate']:.2%}")

    rng = np.random.default_rng(0)
    confs = rng.beta(8, 2, size=2000)
    corrects = (rng.random(2000) < confs).astype(float)
    print("\n=== Calibration Report (simulated production) ===")
    print(full_report(confs, corrects))


if __name__ == "__main__":
    main()
