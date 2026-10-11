"""Train a tiny RLCD model on synthetic choice data; show calibration
before/after and curriculum progression."""
import numpy as np
import torch
from types import SimpleNamespace

from rlcd.core.model import RLCDDecisionModel
from rlcd.core.reward import AdaptiveRewardFunction, DOMAIN_PROFILES
from rlcd.calibration.temperature import AdaptiveTemperatureCalibrator
from rlcd.training.trainer import RLCDTrainConfig, RLCDTrainer
from rlcd.evaluation.metrics import full_report


def make_batch(B=32, T=32, C=4, vocab=1000, seed=None):
    rng = np.random.default_rng(seed)
    tokens = torch.randint(1, vocab, (B, T))
    targets = torch.randint(0, C, (B,))
    wrong = (targets + 1 + torch.randint(0, C - 1, (B,))) % C
    outcomes = torch.tensor((rng.random(B) < 0.8).astype(np.float32))
    batch = {"tokens": tokens, "num_options": C, "targets": targets,
             "wrong_targets": wrong, "outcomes": outcomes}
    texts = [f"synthetic state {i}" for i in range(B)]
    return batch, texts


def collect_calibration(model, steps=20):
    model.eval()
    confs, corrects = [], []
    with torch.no_grad():
        for s in range(steps):
            batch, _ = make_batch(seed=s)
            out, _ = model(batch["tokens"], batch["num_options"])
            probs = out["choice_logits"].softmax(-1)
            conf, pred = probs.max(-1)
            confs += conf.tolist()
            corrects += (pred == batch["targets"]).float().tolist()
    return np.array(confs), np.array(corrects)


def main():
    torch.manual_seed(0)
    cfg = SimpleNamespace(hidden_dim=128, num_layers=2, num_heads=4,
                          vocab_size=1000, max_state_length=64,
                          num_experts=2, num_experts_per_token=1, dropout=0.1)
    model = RLCDDecisionModel(cfg)
    trainer = RLCDTrainer(
        model, AdaptiveRewardFunction(domain_profile=DOMAIN_PROFILES["general"]),
        AdaptiveTemperatureCalibrator(), RLCDTrainConfig(ppo_epochs=2, stage_epochs=2))

    print("BEFORE TRAINING:", full_report(*collect_calibration(model)))
    model.train()
    for epoch in range(6):
        batch, texts = make_batch(seed=100 + epoch)
        m = trainer.train_step(batch, texts)
        print(f"epoch {epoch}: loss={m['loss']:.4f} acc={m['accuracy']:.3f} "
              f"brier={m['brier']:.4f} stage={m['stage']}")
    print("AFTER TRAINING: ", full_report(*collect_calibration(model)))


if __name__ == "__main__":
    main()
