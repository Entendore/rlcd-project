"""RLCD Inference Engine: parallel non-autoregressive answering, hybrid
fast/standard/deep pipeline, LRU cache, confidence-gated routing, INT8."""
from __future__ import annotations
import hashlib, time, uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from rlcd.core.model import RLCDDecisionModel, CrossModalStateProcessor
from rlcd.core.primitives import (Answer, PrimitiveType, Question,
                                  choice_confidence_from_distribution,
                                  noul_confidence_from_probability,
                                  score_confidence_from_distribution)
from rlcd.calibration.temperature import AdaptiveTemperatureCalibrator
from rlcd.calibration.uncertainty import UncertaintyDecomposer
from rlcd.calibration.monitor import CalibrationCache


@dataclass
class RoutingThresholds:
    auto_route: float = 0.85
    auto_with_review: float = 0.65

    def route(self, confidence: float, uncertainty_total: float = 0.0) -> str:
        eff = confidence - 0.3 * uncertainty_total
        if eff >= self.auto_route:
            return "auto_route"
        if eff >= self.auto_with_review:
            return "auto_route_with_review"
        return "manual_review"


class ModelQuantizer:
    @staticmethod
    def quantize(model: RLCDDecisionModel, mode: str = "int8"):
        if mode == "none":
            return model
        model.eval()
        try:
            return torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8)
        except Exception:
            return model


class RLCDInferenceEngine:
    def __init__(self, model, calibrator, uncertainty, cache=None,
                 thresholds=None, fast_path_conf=0.90, deep_path_conf=0.60,
                 quantize_mode="int8", domain="general"):
        self.model = model.eval()
        self.calibrator = calibrator
        self.uncertainty = uncertainty
        self.cache = cache or CalibrationCache()
        self.thresholds = thresholds or RoutingThresholds()
        self.fast_conf, self.deep_conf = fast_path_conf, deep_path_conf
        self.domain = domain
        self._reference_max_probs = np.full(128, 0.85)
        self._fusion = None
        self.model_fast = (ModelQuantizer.quantize(model, quantize_mode)
                           if quantize_mode != "none" else model)

    @staticmethod
    def hash_state(state: str) -> str:
        return hashlib.sha256(state.encode()).hexdigest()[:16]

    @torch.no_grad()
    def answer(self, state: str, questions: List[Question],
               tokens: Optional[torch.Tensor] = None, num_options: int = 8,
               state_embedding: Optional[torch.Tensor] = None,
               image_embedding: Optional[torch.Tensor] = None,
               structured_embedding: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        t0 = time.time()
        state_hash = self.hash_state(state)
        answers: Dict[str, Answer] = {}

        if state_embedding is not None:
            if self._fusion is None:
                hidden = getattr(self.model, "cfg").hidden_dim
                self._fusion = CrossModalStateProcessor(out_dim=hidden)
            fused = self._fusion(state_embedding, image_embedding, structured_embedding)
            out = self.model.heads(fused, num_options)
        else:
            out, _ = self.model(tokens, num_options, pad_mask=None)

        for q in questions:
            cached = self.cache.get(state_hash, q.hash_key())
            if cached is not None:
                answers[q.id] = Answer(**cached["answer"])
                continue

            if q.type == PrimitiveType.CHOICE:
                n_opt = min(len(q.criteria), out["choice_logits"].shape[-1])
                raw_logits = out["choice_logits"][:, :n_opt]
                probs_t = self.calibrator.calibrate_choice(raw_logits, state, self.domain)
                probs = probs_t.squeeze(0).cpu().numpy()
                options = list(q.criteria.keys())[:n_opt]
                pdist = {opt: float(probs[i]) for i, opt in enumerate(options)}
                conf = choice_confidence_from_distribution(pdist)
                value, pr = max(pdist, key=pdist.get), pdist

            elif q.type == PrimitiveType.SCORE:
                lo, hi = q.scale[0], q.scale[-1]
                mean = torch.sigmoid(out["score_mean"]).item()
                value = float(np.clip(lo + (hi - lo) * mean, lo, hi))
                dist = out["score_logits"].softmax(-1).squeeze(0).tolist()
                conf = score_confidence_from_distribution(dist)
                pr = None

            else:  # NOUL
                p_true = self.calibrator.calibrate_noul(
                    out["noul_logit"], state, self.domain).item()
                value, conf, pr = float(p_true), noul_confidence_from_probability(p_true), None

            ans = Answer(question_id=q.id, type=q.type, value=value,
                         probabilities=pr, confidence=conf)

            if conf >= self.fast_conf:
                path = "fast"
            else:
                path = "deep" if conf < self.deep_conf else "standard"
                ans.uncertainty = self.uncertainty.decompose(
                    tokens, num_options,
                    reference_max_probs=(self._reference_max_probs
                                         if path == "deep" else None))

            routing = self.thresholds.route(ans.confidence, ans.uncertainty.total)
            ans.explanation = f"path={path}; routing={routing}; domain={self.domain}"
            self.cache.put(state_hash, q.hash_key(), {"answer": ans.__dict__})
            answers[q.id] = ans

        return {"answers": {k: a.to_dict() for k, a in answers.items()},
                "latency_ms": round((time.time() - t0) * 1000, 2),
                "cache_stats": self.cache.stats(),
                "request_id": str(uuid.uuid4())}

    @torch.no_grad()
    def map_over_dataset(self, states: List[str], questions: List[Question],
                         tokens_list: Optional[List[torch.Tensor]] = None,
                         num_options: int = 8, batch_size: int = 32) -> List[Dict]:
        results = []
        for i, s in enumerate(states):
            tok = tokens_list[i] if tokens_list is not None else None
            results.append(self.answer(s, questions, tokens=tok, num_options=num_options))
        return results
