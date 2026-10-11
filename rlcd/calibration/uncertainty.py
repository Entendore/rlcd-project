"""Uncertainty decomposition: aleatoric (MC-Dropout), epistemic (ensemble),
distributional (OOD score vs training reference)."""
from __future__ import annotations
from typing import List, Optional
import numpy as np
import torch

from rlcd.core.model import RLCDDecisionModel
from rlcd.core.primitives import UncertaintyBreakdown


class UncertaintyDecomposer:
    def __init__(self, model: RLCDDecisionModel,
                 ensemble_models: Optional[List[RLCDDecisionModel]] = None,
                 mc_samples: int = 10):
        self.model = model
        self.ensemble = ensemble_models or []
        self.mc_samples = mc_samples

    @torch.no_grad()
    def aleatoric_uncertainty(self, tokens, num_options) -> float:
        self.model.enable_mc_dropout(self.model)
        preds = []
        for _ in range(self.mc_samples):
            out, _ = self.model(tokens, num_options)
            t = out["noul_logit"].sigmoid()
            preds.append(t.item() if t.dim() == 0 else t.mean().item())
        self.model.disable_mc_dropout(self.model)
        return float(np.std(preds))

    @torch.no_grad()
    def epistemic_uncertainty(self, tokens, num_options) -> float:
        if not self.ensemble:
            return 0.0
        dists = []
        for m in self.ensemble:
            out, _ = m(tokens, num_options)
            dists.append(out["choice_logits"].softmax(-1).squeeze(0).cpu().numpy())
        dists = np.stack(dists)
        mean = dists.mean(0, keepdims=True)
        js = 0.5 * (dists * np.log((dists + 1e-9) / (mean + 1e-9))).sum(-1)
        return float(np.clip(js.mean(), 0, 1))

    @torch.no_grad()
    def distributional_uncertainty(self, tokens, num_options,
                                   reference_max_probs: np.ndarray) -> float:
        out, _ = self.model(tokens, num_options)
        max_p = out["choice_logits"].softmax(-1).max().item()
        ref_q25 = np.percentile(reference_max_probs, 25)
        return float(np.clip((ref_q25 - max_p) / max(ref_q25, 1e-6), 0, 1))

    def decompose(self, tokens, num_options,
                  reference_max_probs: Optional[np.ndarray] = None) -> UncertaintyBreakdown:
        a = self.aleatoric_uncertainty(tokens, num_options)
        e = self.epistemic_uncertainty(tokens, num_options)
        d = (self.distributional_uncertainty(tokens, num_options, reference_max_probs)
             if reference_max_probs is not None else 0.0)
        s = a + e + d + 1e-9
        return UncertaintyBreakdown(aleatoric=a / s, epistemic=e / s, distributional=d / s)
