"""Strictly causal VeriComp model for Next-Token training.

This module is intentionally separate from ``vericomp.modeling``.  The
original model keeps its full-sequence quota allocator and decoder unchanged;
this model uses a streaming-style prefix router and a causal bottleneck for
autoregressive training.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import VeriCompConfig
from .modeling import (
    DNAEmbeddingAndShallowEncoder,
    TransformerBottleneckBackbone,
    quota_js_divergence,
)


class CausalPrefixRouter(nn.Module):
    """Causal adaptive-rate router based only on prefix states.

    The router emits a positive score at every position.  A weighted prefix
    state is sampled at ``N=min(L, max_tokens)`` monotonically increasing
    boundaries.  Position ``t`` can only gather the boundary state whose
    endpoint is at or before ``t``; no future hidden state is ever used.
    """

    def __init__(self, config: VeriCompConfig):
        super().__init__()
        self.max_tokens = config.max_tokens
        self.projection = nn.Linear(config.shallow_dim, config.d_model)
        self.score_conv1 = nn.Conv1d(config.shallow_dim, 128, 7, padding=0)
        self.score_conv2 = nn.Conv1d(128, 1, 1)

    def scores(self, hidden: Tensor) -> Tensor:
        # Left padding makes score[t] depend on hidden[:t+1] only.
        x = F.pad(hidden.transpose(1, 2), (6, 0))
        x = F.gelu(self.score_conv1(x))
        return F.softplus(self.score_conv2(x)).squeeze(1) + 1e-4

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = hidden.shape
        n_tokens = min(length, self.max_tokens)
        scores = self.scores(hidden).float()
        weighted = (hidden.float() * scores.unsqueeze(-1)).cumsum(dim=1)
        normalizer = scores.cumsum(dim=1).unsqueeze(-1).clamp_min(1e-6)
        prefix = (weighted / normalizer).to(hidden.dtype)

        endpoints = torch.div(
            torch.arange(n_tokens, device=hidden.device) * length,
            n_tokens,
            rounding_mode="floor",
        ).long()
        compressed = self.projection(prefix[:, endpoints])
        position_to_token = torch.div(
            torch.arange(length, device=hidden.device) * n_tokens,
            length,
            rounding_mode="floor",
        ).long().clamp_max(n_tokens - 1)
        return compressed, position_to_token, scores.to(hidden.dtype)


class CausalVeriCompNextTokenModel(nn.Module):
    """Causal compression model used only by the Next-Token objective.

    ``local_skip`` is an optional causal residual path for experiments.  It
    uses the full-resolution Mamba prefix state, which is already causal, to
    restore short-range information discarded when many bases share one
    compressed token.  It is disabled by default and is not used by the MLM
    model.
    """

    def __init__(self, config: Optional[VeriCompConfig] = None, *,
                 local_skip: bool = False,
                 local_skip_gate_bias: float = -2.0):
        super().__init__()
        self.config = config or VeriCompConfig(causal_attention=True)
        self.config.causal_attention = True
        self.local_skip = local_skip
        self.encoder = DNAEmbeddingAndShallowEncoder(self.config)
        self.router = CausalPrefixRouter(self.config)
        self.bottleneck = TransformerBottleneckBackbone(self.config)
        if local_skip:
            self.local_skip_norm = nn.LayerNorm(self.config.shallow_dim)
            self.local_skip_proj = nn.Linear(self.config.shallow_dim,
                                             self.config.d_model)
            self.local_skip_gate = nn.Linear(self.config.shallow_dim, 1)
            nn.init.constant_(self.local_skip_gate.bias, local_skip_gate_bias)
        self.lm_head = nn.Linear(self.config.d_model,
                                 self.config.output_vocab_size)

    def forward(self, input_ids_pos: Tensor,
                input_ids_rc: Optional[Tensor] = None,
                labels: Optional[Tensor] = None,
                *, return_bottleneck: bool = False,
                return_decoder_feature: bool = False,
                return_base_feature: bool = False):
        if return_decoder_feature:
            raise ValueError("causal VeriComp has no decoder feature; use feature=backbone")
        high_pos = self.encoder(input_ids_pos)
        compressed, position_to_token, score_pos = self.router(high_pos)
        z_out, loss_router = self.bottleneck(compressed)
        if return_bottleneck:
            return {
                "bottleneck": z_out,
                "loss_router": loss_router,
            }
        gathered = z_out[:, position_to_token]
        if self.local_skip:
            local = self.local_skip_proj(self.local_skip_norm(high_pos))
            gate = torch.sigmoid(self.local_skip_gate(high_pos))
            gathered = gathered + local * gate
        if return_base_feature:
            return {
                "base_feature": gathered,
                "bottleneck": z_out,
                "loss_router": loss_router,
                "position_to_token": position_to_token,
            }
        logits = self.lm_head(gathered)

        loss_strand = logits.new_zeros(())
        score_rc = None
        if input_ids_rc is not None:
            with torch.no_grad():
                high_rc = self.encoder(input_ids_rc)
                score_rc = self.router.scores(high_rc)
                loss_strand = quota_js_divergence(score_pos, score_rc)

        loss_next = logits.new_zeros(())
        if labels is not None:
            loss_next = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
        loss_moe = self.config.router_aux_loss_coef * loss_router.to(loss_next.dtype)
        return {
            "logits": logits,
            "loss": loss_next + loss_moe,
            "loss_mlm": loss_next,
            "loss_next_token": loss_next,
            "loss_router": loss_router,
            "loss_moe": loss_moe,
            "loss_strand": loss_strand,
            "b_pos": score_pos,
            "b_rc": score_rc,
            "position_to_token": position_to_token,
        }

    def num_parameters(self, trainable_only: bool = True) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters
                          if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)
