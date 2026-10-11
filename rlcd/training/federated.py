"""Federated RLCD: privacy-preserving collaborative calibration.
Only the typed output heads are aggregated; backbones stay local."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
import torch.nn.functional as F

from rlcd.core.model import RLCDDecisionModel


@dataclass
class ClientUpdate:
    client_id: str
    head_weights: dict
    n_samples: int
    local_ece: float


class FederatedRLCDServer:
    def __init__(self, global_model: RLCDDecisionModel,
                 byzantine_tolerance: float = 0.2, min_clients: int = 3):
        self.global_model = global_model
        self.byz_frac = byzantine_tolerance
        self.min_clients = min_clients
        self.round = 0

    def _secure_aggregate(self, updates: List[ClientUpdate]):
        eces = np.array([u.local_ece for u in updates])
        k = int(len(updates) * self.byz_frac)
        keep_idx = (np.argsort(eces)[k:len(updates) - k] if k
                    else np.arange(len(updates)))
        kept = [updates[i] for i in keep_idx]
        w_total = sum(u.n_samples for u in kept)
        global_head = self.global_model.heads.state_dict()
        for key in global_head:
            acc = torch.zeros_like(global_head[key], dtype=torch.float32)
            for u in kept:
                acc += (u.n_samples / w_total) * u.head_weights[key].float()
            global_head[key] = acc
        self.global_model.heads.load_state_dict(global_head)

    def federated_round(self, client_updates: List[ClientUpdate]):
        assert len(client_updates) >= self.min_clients
        self._secure_aggregate(client_updates)
        self.round += 1
        return {"round": self.round, "clients": len(client_updates)}


class FederatedRLCDClient:
    def __init__(self, client_id: str, model: RLCDDecisionModel):
        self.client_id = client_id
        self.model = model

    def local_calibrate(self, dataloader, epochs: int = 1) -> ClientUpdate:
        self.model.freeze_backbone()
        self.model.heads.train()
        opt = torch.optim.AdamW(self.model.heads.parameters(), lr=1e-4)
        n, correct, confs = 0, 0, []
        for _ in range(epochs):
            for batch in dataloader:
                tokens, nopt, targets = batch["tokens"], batch["num_options"], batch["targets"]
                out, _ = self.model(tokens, nopt)
                loss = F.cross_entropy(out["choice_logits"], targets)
                opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    p = out["choice_logits"].softmax(-1)
                    confs.extend(p.max(-1)[0].tolist())
                    correct += (p.argmax(-1) == targets).sum().item()
                    n += len(targets)
        local_ece = abs(correct / max(n, 1) - float(np.mean(confs)))
        head_state = {k: v.clone() for k, v in self.model.heads.state_dict().items()}
        return ClientUpdate(self.client_id, head_state, n, local_ece)

    def adopt_global(self, server: FederatedRLCDServer):
        self.model.heads.load_state_dict(server.global_model.heads.state_dict())
        self.model.unfreeze_backbone()
