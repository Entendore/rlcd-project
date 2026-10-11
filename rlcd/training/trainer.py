"""RLCD Trainer: PPO + adaptive rewards + Contrastive Calibration Learning +
Adversarial Calibration Training (PGD) + Curriculum Learning."""
from __future__ import annotations
import copy, logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlcd.core.model import RLCDDecisionModel
from rlcd.core.reward import AdaptiveRewardFunction, DomainRiskProfile, DOMAIN_PROFILES
from rlcd.calibration.temperature import AdaptiveTemperatureCalibrator

log = logging.getLogger("rlcd.trainer")


@dataclass
class RLCDTrainConfig:
    lr: float = 3e-4
    ppo_epochs: int = 4
    ppo_clip: float = 0.2
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    contrastive_margin: float = 0.1
    contrastive_weight: float = 0.3
    adv_epsilon: float = 0.05
    adv_alpha: float = 0.01
    adv_steps: int = 7
    adv_weight: float = 0.5
    curriculum_stages: int = 3
    stage_epochs: int = 10


class ContrastiveCalibrationLoss(nn.Module):
    """Correct answers well-calibrated; wrong answers pushed below by margin."""

    def __init__(self, margin: float = 0.1, temperature: float = 0.07):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, correct_conf, wrong_conf):
        gap = F.relu(self.margin - (correct_conf - wrong_conf))
        overconf = wrong_conf.pow(2).mean()
        underconf = (1.0 - correct_conf).pow(2).mean()
        return gap.mean() + 0.5 * overconf + 0.5 * underconf


class AdversarialCalibrationTrainer:
    """PGD attacks in embedding space that MAXIMIZE calibration error."""

    def __init__(self, model: RLCDDecisionModel, cfg: RLCDTrainConfig):
        self.model = model
        self.cfg = cfg

    @staticmethod
    def _calibration_loss(probs, targets):
        p = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return ((p - 1.0) ** 2).mean()

    def generate_adversarial_embeddings(self, pooled, num_options, targets):
        delta = torch.zeros_like(pooled, requires_grad=True)
        for _ in range(self.cfg.adv_steps):
            pooled_adv = pooled + delta
            logits = self.model.heads.choice_head(pooled_adv)[:, :num_options]
            probs = logits.softmax(-1)
            loss = -self._calibration_loss(probs, targets)
            loss.backward()
            with torch.no_grad():
                delta.data = delta.data + self.cfg.adv_alpha * delta.grad.sign()
                delta.data = delta.data.clamp(-self.cfg.adv_epsilon, self.cfg.adv_epsilon)
            delta.grad.zero_()
        return (pooled + delta.data).detach()


class CurriculumScheduler:
    STAGES = ["easy", "ambiguous", "robust"]
    STAGE_BETA_MULT = {"easy": 0.7, "ambiguous": 1.0, "robust": 1.3}
    STAGE_NOISE = {"easy": 0.0, "ambiguous": 0.05, "robust": 0.15}

    def __init__(self, stages: int = 3, epochs_per_stage: int = 10):
        self.stages, self.epochs_per_stage = stages, epochs_per_stage
        self.current_stage, self.epoch_in_stage = 0, 0

    def step_epoch(self) -> bool:
        self.epoch_in_stage += 1
        if (self.epoch_in_stage >= self.epochs_per_stage
                and self.current_stage < self.stages - 1):
            self.current_stage += 1
            self.epoch_in_stage = 0
            log.info("Curriculum -> stage %d (%s)", self.current_stage, self.STAGES[self.current_stage])
            return True
        return False

    @property
    def stage_name(self): return self.STAGES[self.current_stage]

    @property
    def beta_multiplier(self): return self.STAGE_BETA_MULT[self.stage_name]

    @property
    def noise_level(self): return self.STAGE_NOISE[self.stage_name]


class RLCDTrainer:
    def __init__(self, model, reward_fn, calibrator, cfg: RLCDTrainConfig,
                 domain_profile: Optional[DomainRiskProfile] = None):
        self.model = model
        self.ref_model = copy.deepcopy(model).eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.reward_fn = reward_fn
        self.calibrator = calibrator
        self.cfg = cfg
        self.domain = domain_profile or DOMAIN_PROFILES["general"]
        self.contrastive = ContrastiveCalibrationLoss(cfg.contrastive_margin)
        self.adversarial = AdversarialCalibrationTrainer(model, cfg)
        self.curriculum = CurriculumScheduler(cfg.curriculum_stages, cfg.stage_epochs)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
        self.ensemble_models: List[RLCDDecisionModel] = []
        self.history: List[Dict] = []

    def _ppo_update(self, tokens, num_options, targets, old_log_probs, advantages):
        for _ in range(self.cfg.ppo_epochs):
            out, aux = self.model(tokens, num_options)
            dist = torch.distributions.Categorical(logits=out["choice_logits"])
            new_log_probs = dist.log_prob(targets)
            ratio = torch.exp(new_log_probs - old_log_probs)
            s1 = ratio * advantages
            s2 = torch.clamp(ratio, 1 - self.cfg.ppo_clip, 1 + self.cfg.ppo_clip) * advantages
            policy_loss = -torch.min(s1, s2).mean()
            loss = (policy_loss - self.cfg.entropy_coef * dist.entropy().mean()
                    + 0.01 * aux)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()

    def train_step(self, batch: Dict[str, torch.Tensor],
                   state_texts: List[str]) -> Dict[str, float]:
        tokens = batch["tokens"]
        num_options = batch["num_options"]
        targets = batch["targets"]
        wrong_targets = batch["wrong_targets"]
        outcomes = batch["outcomes"].float()
        stage_beta = self.curriculum.beta_multiplier

        with torch.no_grad():
            old_out, _ = self.model(tokens, num_options)
            old_logits = old_out["choice_logits"]
            old_dist = torch.distributions.Categorical(logits=old_logits)
            old_log_probs = old_dist.log_prob(targets)
            old_probs = old_logits.softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        rewards = []
        for i, txt in enumerate(state_texts):
            r = self.reward_fn.compute(
                pred_correct=(old_probs[i] > 0.5).float().unsqueeze(0),
                pred_prob=old_probs[i].unsqueeze(0),
                outcome=outcomes[i].unsqueeze(0),
                neighbor_probs=None,
                state_text=txt, num_options=num_options)
            rewards.append(r["reward"].item())
        advantages = torch.tensor(rewards, device=tokens.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self._ppo_update(tokens, num_options, targets, old_log_probs, advantages)

        out, aux = self.model(tokens, num_options)
        probs = out["choice_logits"].softmax(-1)
        correct_conf = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        wrong_conf = probs.gather(-1, wrong_targets.unsqueeze(-1)).squeeze(-1)
        ccl_loss = self.contrastive(correct_conf, wrong_conf)

        p_true = correct_conf.clamp(1e-9, 1 - 1e-9)
        log_score = -(outcomes * torch.log(p_true)
                      + (1 - outcomes) * torch.log(1 - p_true)).mean()
        brier = ((correct_conf - outcomes) ** 2).mean()

        with torch.no_grad():
            self.calibrator.fit(logits.detach().cpu(), targets.cpu(),
                                complexities=np.array(
                                    [self.reward_fn.get_weights(t, num_options)["complexity"]
                                     for t in state_texts]),
                                domains=[self.domain.name] * len(state_texts))

        calib_loss = (ccl_loss + log_score + brier) * stage_beta

        adv_loss = torch.tensor(0.0)
        if self.curriculum.current_stage >= 1:
            pooled = self.model.backbone(tokens)[0].mean(dim=1)
            pooled_adv = self.adversarial.generate_adversarial_embeddings(
                pooled, num_options, targets)
            adv_probs = self.model.heads.choice_head(pooled_adv)[:, :num_options].softmax(-1)
            ap = adv_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).clamp(1e-9, 1 - 1e-9)
            adv_loss = -(outcomes * torch.log(ap)
                         + (1 - outcomes) * torch.log(1 - ap)).mean()
            adv_loss = adv_loss * self.cfg.adv_weight * stage_beta

        total_loss = calib_loss + adv_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        self.curriculum.step_epoch()
        metrics = {"loss": total_loss.item(), "ccl_loss": ccl_loss.item(),
                   "brier": brier.item(), "log_score": log_score.item(),
                   "adv_loss": float(adv_loss), "stage": self.curriculum.stage_name,
                   "mean_confidence": correct_conf.mean().item(),
                   "accuracy": (probs.argmax(-1) == targets).float().mean().item()}
        self.history.append(metrics)
        return metrics

    def distill_to_ensemble(self, dataloader, n_members: int = 5):
        """Train diverse ensemble members to power epistemic uncertainty."""
        members = []
        for i in range(n_members):
            m = copy.deepcopy(self.model)
            torch.manual_seed(1000 + i)
            for p in m.parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p) * 1e-3)
            opt = torch.optim.AdamW(m.parameters(), lr=self.cfg.lr * 0.5)
            m.train()
            for batch in dataloader:
                out, _ = m(batch["tokens"], batch["num_options"])
                loss = F.cross_entropy(out["choice_logits"], batch["targets"])
                opt.zero_grad(); loss.backward(); opt.step()
            m.eval()
            members.append(m)
        self.ensemble_models = members
        return members
