"""Jev-style System One model: transformer backbone + MoE + typed decision heads."""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEFeedForward(nn.Module):
    """Sparse Mixture-of-Experts FFN (top-k routing + load-balance aux loss)."""

    def __init__(self, hidden_dim: int, num_experts: int = 8,
                 top_k: int = 2, expert_mult: float = 4.0):
        super().__init__()
        self.num_experts, self.top_k = num_experts, top_k
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, int(hidden_dim * expert_mult)),
                          nn.GELU(),
                          nn.Linear(int(hidden_dim * expert_mult), hidden_dim))
            for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, H = x.shape
        flat = x.reshape(-1, H)
        logits = self.gate(flat)
        weights, idx = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        out = torch.zeros_like(flat)
        load = torch.zeros(self.num_experts, device=x.device)
        for e, expert in enumerate(self.experts):
            mask = (idx == e)
            if not mask.any():
                continue
            sel = mask.any(dim=-1)
            w = (weights * mask).sum(dim=-1)[sel].unsqueeze(-1)
            out[sel] += w * expert(flat[sel])
            load[e] = sel.float().mean()
        importance = logits.softmax(-1).mean(0)
        aux_loss = self.num_experts * (importance * load).sum()
        return out.reshape(B, T, H), aux_loss


class DecisionBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden_dim
        self.ln1 = nn.LayerNorm(h)
        self.attn = nn.MultiheadAttention(h, cfg.num_heads,
                                          dropout=cfg.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(h)
        self.moe = MoEFeedForward(h, cfg.num_experts, cfg.num_experts_per_token)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, pad_mask=None):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=pad_mask, need_weights=False)
        x = x + self.dropout(attn_out)
        ff, aux = self.moe(self.ln2(x))
        return x + self.dropout(ff), aux


class TransformerBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.pos_emb = nn.Embedding(cfg.max_state_length, cfg.hidden_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([DecisionBlock(cfg) for _ in range(cfg.num_layers)])
        self.ln_f = nn.LayerNorm(cfg.hidden_dim)
        self.register_buffer("position_ids",
                             torch.arange(cfg.max_state_length).unsqueeze(0),
                             persistent=False)

    def forward(self, tokens, pad_mask=None):
        B, T = tokens.shape
        pos = self.position_ids[:, :T]
        x = self.drop(self.tok_emb(tokens) + self.pos_emb(pos))
        total_aux = 0.0
        for blk in self.blocks:
            x, aux = blk(x, pad_mask)
            total_aux = total_aux + aux
        return self.ln_f(x), total_aux


class CrossModalStateProcessor(nn.Module):
    """Fuses text, image and structured-data embeddings into one state vector."""

    def __init__(self, text_dim=768, image_dim=512, structured_dim=128, out_dim=768):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, out_dim)
        self.image_proj = nn.Linear(image_dim, out_dim)
        self.structured_proj = nn.Linear(structured_dim, out_dim)
        self.t2i_attn = nn.MultiheadAttention(out_dim, 8, batch_first=True)
        self.i2t_attn = nn.MultiheadAttention(out_dim, 8, batch_first=True)
        self.fuse = nn.Sequential(nn.Linear(out_dim * 3, out_dim * 2), nn.GELU(),
                                  nn.Linear(out_dim * 2, out_dim),
                                  nn.LayerNorm(out_dim))

    def forward(self, text_emb, image_emb=None, structured_emb=None):
        t = self.text_proj(text_emb)
        i = self.image_proj(image_emb) if image_emb is not None else torch.zeros_like(t)
        s = (self.structured_proj(structured_emb) if structured_emb is not None
             else torch.zeros_like(t))
        t2, _ = self.t2i_attn(t, i, i)
        i2, _ = self.i2t_attn(i, t, t)
        return self.fuse(torch.cat([t + t2, i + i2, s], dim=-1))


class TypedOutputHeads(nn.Module):
    """Specialized heads per primitive + confidence-estimation branch."""

    def __init__(self, cfg, max_options: int = 255, score_bins: int = 33):
        super().__init__()
        h = cfg.hidden_dim
        self.choice_head = nn.Linear(h, max_options)
        self.score_mean = nn.Linear(h, 1)
        self.score_dist = nn.Linear(h, score_bins)
        self.noul_head = nn.Linear(h, 1)
        self.confidence_head = nn.Sequential(
            nn.Linear(h, h // 2), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(h // 2, 1), nn.Sigmoid())

    def forward(self, pooled: torch.Tensor, num_options: int) -> Dict[str, torch.Tensor]:
        return {
            "choice_logits": self.choice_head(pooled)[:, :num_options],
            "score_mean": self.score_mean(pooled).squeeze(-1),
            "score_logits": self.score_dist(pooled),
            "noul_logit": self.noul_head(pooled).squeeze(-1),
            "confidence": self.confidence_head(pooled).squeeze(-1),
        }


class RLCDDecisionModel(nn.Module):
    """Non-autoregressive: ALL question outputs produced in one parallel pass."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = TransformerBackbone(cfg)
        self.heads = TypedOutputHeads(cfg)

    @staticmethod
    def enable_mc_dropout(model: nn.Module):
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    @staticmethod
    def disable_mc_dropout(model: nn.Module):
        model.eval()

    def _pool(self, hidden, pad_mask):
        if pad_mask is not None:
            keep = (~pad_mask).unsqueeze(-1).float()
            return (hidden * keep).sum(1) / keep.sum(1).clamp(min=1)
        return hidden.mean(dim=1)

    def forward(self, tokens, num_options, pad_mask=None):
        hidden, aux = self.backbone(tokens, pad_mask)
        pooled = self._pool(hidden, pad_mask)
        return self.heads(pooled, num_options), aux

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def num_parameters(self, trainable_only=False) -> int:
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad or not trainable_only)
