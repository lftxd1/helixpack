"""VeriComp-100M model.

The allocator is implemented from the same triangular kernel as the PRD.  A
dense W_norm is materialized for small inputs.  For long inputs, its exactly
equivalent three-neighbour representation is retained, which avoids an
otherwise unavoidable [B, 4096, 1048576] allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from mamba_ssm import Mamba2

from .config import VeriCompConfig


def quota_js_divergence(b_pos: Tensor, b_rc: Tensor) -> Tensor:
    """Symmetric JS divergence between forward and aligned quota fields."""
    p = b_pos.float()
    q = torch.flip(b_rc.float(), dims=[-1])
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    midpoint = 0.5 * (p + q)
    midpoint_log = midpoint.clamp_min(1e-8).log()
    return 0.5 * (
        (p * (p.clamp_min(1e-8).log() - midpoint_log)).sum(dim=-1).mean()
        + (q * (q.clamp_min(1e-8).log() - midpoint_log)).sum(dim=-1).mean()
    )


@dataclass
class DualDynamicWeights:
    """Lazy equivalent of W_norm for ultra-long sequences.

    ``shape`` is the dense matrix shape. ``to_dense`` is intentionally
    explicit because materializing a 1M sequence matrix is very expensive.
    """

    cumulative: Tensor
    denominator: Tensor
    n_tokens: int
    dense: Optional[Tensor] = None

    @property
    def shape(self):
        b, length = self.cumulative.shape
        return (b, self.n_tokens, length)

    def to_dense(self, chunk_size: int = 64) -> Tensor:
        if self.dense is not None:
            return self.dense
        b, length = self.cumulative.shape
        out = self.cumulative.new_empty((b, self.n_tokens, length))
        for start in range(0, self.n_tokens, chunk_size):
            stop = min(start + chunk_size, self.n_tokens)
            grid = torch.arange(start + 1, stop + 1, device=self.cumulative.device,
                                dtype=self.cumulative.dtype)
            raw = (1.0 - (self.cumulative[:, None, :] - grid[None, :, None]).abs()).clamp_min(0)
            out[:, start:stop] = raw / self.denominator[:, start:stop, None].clamp_min(torch.finfo(raw.dtype).eps)
        self.dense = out
        return out


@dataclass
class GridWeights:
    """Fixed-stride block-mean allocation, kept symbolic.

    Row ``i`` of the equivalent dense ``W_norm`` holds exactly ``stride``
    equal entries of ``1/stride`` over base positions
    ``[i*stride, (i+1)*stride)``.  Storing the rule instead of the matrix
    keeps a 1M-base input from materializing a ``n_tokens x length`` tensor.
    """

    n_tokens: int
    stride: int
    length: int
    batch: int
    device: torch.device = torch.device("cpu")
    dense: Optional[Tensor] = None

    @property
    def shape(self):
        return (self.batch, self.n_tokens, self.length)

    def to_dense(self, chunk_size: int = 64) -> Tensor:
        if self.dense is not None:
            return self.dense
        out = torch.zeros((self.batch, self.n_tokens, self.length),
                          device=self.device)
        weight = 1.0 / self.stride
        for start in range(0, self.n_tokens, chunk_size):
            stop = min(start + chunk_size, self.n_tokens)
            rows = torch.arange(start, stop, device=self.device)
            cols = rows[:, None] * self.stride + torch.arange(
                self.stride, device=self.device)[None, :]
            valid = cols < self.length
            cols = cols.clamp(max=max(self.length - 1, 0))
            values = torch.full((stop - start, self.stride), weight,
                                device=self.device) * valid
            out[:, start:stop].scatter_(
                2, cols[None].expand(self.batch, -1, -1),
                values[None].expand(self.batch, -1, -1))
        self.dense = out
        return out


class MambaResidualBlock(nn.Module):
    """GeneZip-style pre-routing Mamba2 block with a residual stream."""

    def __init__(self, config: VeriCompConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.shallow_dim)
        self.mixer = Mamba2(
            d_model=config.shallow_dim,
            d_state=config.shallow_mamba_state,
            d_conv=config.shallow_mamba_conv,
            expand=config.shallow_mamba_expand,
            headdim=config.shallow_mamba_head_dim,
            chunk_size=config.shallow_mamba_chunk_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mixer(self.norm(x))


class DNAEmbeddingAndShallowEncoder(nn.Module):
    """Full-length contextual Mamba2 encoder used before dynamic routing.

    Like GeneZip, this encoder runs before routing and preserves sequence
    length. The resulting contextual states drive boundary estimation, while
    the same full-resolution states remain available as the decoder skip
    connection.
    """

    def __init__(self, config: VeriCompConfig):
        super().__init__()
        self.activation_checkpointing = config.activation_checkpointing
        self.bf16_residual_stream = bool(config.bf16_residual_stream)
        self.embedding = nn.Embedding(config.vocab_size, config.shallow_dim,
                                      padding_idx=config.pad_token_id)
        if config.shallow_mamba_layers <= 0:
            raise ValueError("shallow_mamba_layers must be positive")
        self.blocks = nn.ModuleList([
            MambaResidualBlock(config)
            for _ in range(config.shallow_mamba_layers)
        ])

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embedding(input_ids)
        if self.bf16_residual_stream:
            autocast_bf16 = (
                x.device.type == "cuda"
                and torch.is_autocast_enabled("cuda")
                and torch.get_autocast_dtype("cuda") == torch.bfloat16
            )
            if autocast_bf16 or self.embedding.weight.dtype == torch.bfloat16:
                x = x.to(torch.bfloat16)
        for block in self.blocks:
            if self.training and self.activation_checkpointing:
                x = activation_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


class QuotaDynamicCompressor1M(nn.Module):
    """Length-adaptive, boundary-aware quota compressor.

    The token budget is determined only by the input length,
    ``N(L)=min(L,max_tokens)``. GeneZip-style adjacent-state dissimilarity
    then biases the learned importance scores toward likely boundaries, while
    quota normalization still guarantees exactly N bottleneck tokens.
    """

    def __init__(self, config: VeriCompConfig):
        super().__init__()
        if config.allocation_mode not in {
            "dynamic", "uniform", "length_only", "grid"
        }:
            raise ValueError(
                "allocation_mode must be dynamic, uniform, length_only, or grid"
            )
        self.max_tokens = config.max_tokens
        self.allocation_mode = config.allocation_mode
        self.grid_stride = int(getattr(config, "grid_stride", 32))
        if self.allocation_mode == "grid" and self.grid_stride <= 0:
            raise ValueError("grid_stride must be positive")
        self.strict_uncompressed_short = bool(config.strict_uncompressed_short)
        self.score_net = nn.Sequential(
            nn.Conv1d(config.shallow_dim, 128, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Softplus(),
        )
        self.boundary_q = nn.Linear(config.shallow_dim, config.shallow_dim, bias=False)
        self.boundary_k = nn.Linear(config.shallow_dim, config.shallow_dim, bias=False)
        with torch.no_grad():
            self.boundary_q.weight.copy_(torch.eye(config.shallow_dim))
            self.boundary_k.weight.copy_(torch.eye(config.shallow_dim))
        self.boundary_router_weight = config.boundary_router_weight
        self.projection = nn.Linear(config.shallow_dim, config.d_model)
        self.dense_weight_max_elements = config.dense_weight_max_elements
        self.quota_mask_token_id: Optional[int] = None

    def quota_keep_mask_for(self, input_ids: Tensor, length: int) -> Optional[Tensor]:
        """Keep-mask that zeroes allocation evidence on padding tokens."""
        if self.quota_mask_token_id is None:
            return None
        keep = input_ids.ne(self.quota_mask_token_id)
        if keep.shape[1] != length:
            raise ValueError(
                "quota keep mask length mismatch: "
                f"{keep.shape[1]} vs hidden length {length}"
            )
        return keep

    def target_tokens(self, length: int) -> int:
        """Return the deterministic bottleneck budget for an input length."""
        if length <= 0:
            raise ValueError("sequence length must be positive")
        if self.allocation_mode == "grid":
            # Fixed compression ratio: every ``grid_stride`` bases collapse to
            # one bottleneck token regardless of input length.  This is the
            # deliberate opposite of the length-adaptive rule below.
            return -(-length // self.grid_stride)
        return min(length, self.max_tokens)

    def boundary_scores(self, hidden: Tensor) -> Tensor:
        """Estimate GeneZip-style boundary probability for every base.

        The first base is always treated as a boundary. The computation is
        fully differentiable and has shape [B,L].
        """
        bsz, length, _ = hidden.shape
        if length == 1:
            return hidden.new_ones((bsz, 1))
        q = F.normalize(self.boundary_q(hidden[:, :-1]).float(), dim=-1)
        k = F.normalize(self.boundary_k(hidden[:, 1:]).float(), dim=-1)
        boundary = ((1.0 - (q * k).sum(dim=-1)) * 0.5).clamp(0.0, 1.0)
        return F.pad(boundary.to(hidden.dtype), (1, 0), value=1.0)

    def importance_scores(self, hidden: Tensor) -> Tensor:
        """Combine learned importance with local boundary evidence."""
        learned = self.score_net(hidden.transpose(1, 2)).squeeze(1)
        boundary = self.boundary_scores(hidden)
        return learned * (1.0 + self.boundary_router_weight * boundary)

    def allocation_scores(self, hidden: Tensor) -> Tensor:
        """Return content-dependent or uniform allocation evidence.

        ``uniform`` and ``length_only`` intentionally share the exact same
        interpolation and sparse aggregation path as the dynamic allocator.
        Only the position evidence is replaced by a constant field, which
        makes them controlled compression baselines rather than a second
        decoder implementation.
        """
        if self.allocation_mode == "grid":
            # Grid mode never consults content evidence, and the short-input
            # identity shortcut must not pre-empt it either.
            return hidden.new_ones(hidden.shape[:2])
        if self.strict_uncompressed_short and hidden.shape[1] <= self.max_tokens:
            return hidden.new_ones(hidden.shape[:2])
        if self.allocation_mode in {"uniform", "length_only"}:
            return hidden.new_ones(hidden.shape[:2])
        return self.importance_scores(hidden)

    def scores_to_quota(self, scores: Tensor, n_tokens: int) -> Tensor:
        scores = scores.float()
        return scores / scores.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(scores.dtype).eps) * n_tokens

    @staticmethod
    def _three_neighbours(cumulative: Tensor, n_tokens: int):
        # Mathematical k is one-based. At most k=floor(C)-1..floor(C)+1 can
        # have a non-zero triangular kernel value.
        base = cumulative.floor().long()
        delta = torch.tensor((-1, 0, 1), device=cumulative.device)
        indices = base[:, None, :] + delta[None, :, None] - 1
        valid = (indices >= 0) & (indices < n_tokens)
        valid[:, 1:] &= indices[:, 1:] != indices[:, :-1]
        valid[:, 2:] &= indices[:, 2:] != indices[:, 0:1]
        safe_indices = indices.clamp(0, n_tokens - 1)
        grid = (safe_indices + 1).to(cumulative.dtype)
        raw = (1.0 - (cumulative[:, None, :] - grid).abs()).clamp_min(0)
        raw = raw * valid.to(raw.dtype)
        return safe_indices, raw

    def make_allocation(self, quota: Tensor, n_tokens: int,
                        dense: bool = False) -> DualDynamicWeights | Tensor:
        cumulative = torch.cumsum(quota, dim=-1)
        indices, raw = self._three_neighbours(cumulative, n_tokens)
        bsz, length = cumulative.shape
        denominator = cumulative.new_zeros((bsz, n_tokens))
        for j in range(3):
            denominator.scatter_add_(1, indices[:, j], raw[:, j])

        dense_elements = bsz * n_tokens * length
        if dense or dense_elements <= self.dense_weight_max_elements:
            return self._dense_from_parts(cumulative, denominator, n_tokens)
        return DualDynamicWeights(cumulative, denominator, n_tokens)

    @staticmethod
    def _dense_from_parts(cumulative: Tensor, denominator: Tensor, n_tokens: int,
                          chunk_size: int = 128) -> Tensor:
        bsz, length = cumulative.shape
        out = cumulative.new_empty((bsz, n_tokens, length))
        eps = torch.finfo(cumulative.dtype).eps
        for start in range(0, n_tokens, chunk_size):
            stop = min(start + chunk_size, n_tokens)
            grid = torch.arange(start + 1, stop + 1, device=cumulative.device,
                                dtype=cumulative.dtype)
            raw = (1.0 - (cumulative[:, None, :] - grid[None, :, None]).abs()).clamp_min(0)
            out[:, start:stop] = raw / denominator[:, start:stop, None].clamp_min(eps)
        return out

    @staticmethod
    def _sparse_aggregate(hidden: Tensor, indices: Tensor, raw: Tensor,
                          denominator: Tensor, n_tokens: int) -> Tensor:
        bsz, length, dim = hidden.shape
        # Dense matmul accumulates products at higher precision even when the
        # input/output tensors are BF16.  Match that behavior in the sparse
        # scatter path.  BF16 scatter_add would otherwise introduce large
        # order-dependent errors at long sequence lengths, especially when a
        # bottleneck token pools many bases.
        numerator = torch.zeros(
            (bsz, n_tokens, dim), device=hidden.device, dtype=torch.float32
        )
        for j in range(3):
            values = hidden.float() * raw[:, j, :].float().unsqueeze(-1)
            idx = indices[:, j]
            numerator.scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, dim), values)
        output = numerator / denominator.float().clamp_min(torch.finfo(torch.float32).eps).unsqueeze(-1)
        return output.to(hidden.dtype)

    @staticmethod
    def _grid_pool(hidden: Tensor, stride: int, n_tokens: int,
                   quota_keep_mask: Optional[Tensor] = None) -> Tensor:
        """Average each non-overlapping block of ``stride`` bases."""
        bsz, length, dim = hidden.shape
        padded = n_tokens * stride
        if quota_keep_mask is None:
            if padded == length:
                return hidden.reshape(bsz, n_tokens, stride, dim).mean(dim=2)
            padded_hidden = F.pad(hidden, (0, 0, 0, padded - length))
            return padded_hidden.reshape(bsz, n_tokens, stride, dim).mean(dim=2)
        keep = quota_keep_mask.to(hidden.dtype).unsqueeze(-1)
        if padded != length:
            hidden = F.pad(hidden, (0, 0, 0, padded - length))
            keep = F.pad(keep, (0, 0, 0, padded - length))
        blocks = hidden.reshape(bsz, n_tokens, stride, dim)
        keep = keep.reshape(bsz, n_tokens, stride, 1)
        return (blocks * keep).sum(dim=2) / keep.sum(dim=2).clamp_min(1.0)

    def _forward_grid(self, hidden: Tensor, n_tokens: int,
                      quota_keep_mask: Optional[Tensor]):
        """Fixed-stride block-mean compression.

        No content scores, no boundary evidence, no dual-strand fusion and no
        length-dependent budget: consecutive blocks of ``grid_stride`` bases
        are averaged, so the compression ratio is constant by construction.
        """
        bsz, length, _ = hidden.shape
        pooled = self._grid_pool(hidden, self.grid_stride, n_tokens,
                                 quota_keep_mask)
        compressed = self.projection(pooled)
        allocation = GridWeights(
            n_tokens=n_tokens, stride=self.grid_stride, length=length,
            batch=bsz, device=hidden.device,
        )
        boundaries = torch.arange(1, length + 1, device=hidden.device,
                                  dtype=torch.float32) / self.grid_stride
        self.last_cumulative = boundaries.unsqueeze(0).expand(bsz, -1).detach()
        quota = hidden.new_full((bsz, length), 1.0 / self.grid_stride)
        return compressed, allocation, quota

    def forward(self, hidden: Tensor, *, force_dense: bool = False,
                quota_keep_mask: Optional[Tensor] = None):
        bsz, length, _ = hidden.shape
        n_tokens = self.target_tokens(length)
        if self.allocation_mode == "grid":
            return self._forward_grid(hidden, n_tokens, quota_keep_mask)
        scores = self.allocation_scores(hidden)
        if quota_keep_mask is not None:
            scores = scores * quota_keep_mask.to(scores.dtype)
        quota = self.scores_to_quota(scores, n_tokens)
        allocation = self.make_allocation(quota, n_tokens, dense=force_dense)
        projected = self.projection(hidden)
        if isinstance(allocation, Tensor):
            compressed = torch.bmm(allocation, projected)
            denominator = allocation.sum(dim=-1)
        else:
            indices, raw = self._three_neighbours(allocation.cumulative, n_tokens)
            compressed = self._sparse_aggregate(projected, indices, raw,
                                                allocation.denominator, n_tokens)
            denominator = allocation.denominator
            allocation = DualDynamicWeights(allocation.cumulative, denominator, n_tokens)
        allocation_obj = allocation
        if not isinstance(allocation_obj, Tensor):
            allocation_obj = allocation_obj.cumulative
        self.last_cumulative = allocation_obj.detach()
        b_quota = quota.to(hidden.dtype)
        return compressed, allocation, b_quota

    def forward_pair(self, hidden_pos: Tensor, hidden_rc: Tensor,
                     *, force_dense: bool = False,
                     independent_normalization: bool = False):
        """Compress using symmetric evidence from both DNA strands.

        ``hidden_rc`` is in reverse-complement sequence order.  Its importance
        scores are computed in that native coordinate system first (so its
        native first-base boundary remains correct), then flipped into the
        positive-strand coordinate system.  Each strand proposes importance
        scores independently; the actual quota is their equal-weight mean, so
        neither strand is treated as the authority when local compression
        preferences disagree.

        Returns the positive compressed representation, the shared allocation,
        the shared quota, and the two per-strand quota proposals.  The reverse
        proposal is returned in its original reverse-complement order so the
        existing strand diagnostic can align it explicitly.
        """
        if hidden_pos.shape != hidden_rc.shape:
            raise ValueError(
                "paired strand hidden states must have identical shapes, got "
                f"{tuple(hidden_pos.shape)} and {tuple(hidden_rc.shape)}"
            )
        _, length, _ = hidden_pos.shape
        n_tokens = self.target_tokens(length)
        if self.allocation_mode == "grid":
            # There is no per-strand evidence to reconcile, so the paired path
            # degenerates to the single-strand grid path.  Callers skip the RC
            # encoder entirely in this mode.
            z, allocation, b_quota = self._forward_grid(hidden_pos, n_tokens, None)
            return (z, allocation, b_quota, b_quota,
                    b_quota.flip(1).to(hidden_rc.dtype))
        scores_pos = self.allocation_scores(hidden_pos)
        scores_rc_aligned = self.allocation_scores(hidden_rc).flip(1)
        if independent_normalization:
            # Dual-independent: normalize each strand to the full budget
            # before averaging. This is deliberately different from the
            # production Dual path, which averages raw evidence first and
            # normalizes only once.
            pos_quota = self.scores_to_quota(scores_pos, n_tokens)
            rc_quota_aligned = self.scores_to_quota(scores_rc_aligned, n_tokens)
            joint_quota = 0.5 * (pos_quota + rc_quota_aligned)
        else:
            joint_scores = 0.5 * (scores_pos + scores_rc_aligned)
            joint_quota = self.scores_to_quota(joint_scores, n_tokens)
        allocation = self.make_allocation(joint_quota, n_tokens, dense=force_dense)

        projected = self.projection(hidden_pos)
        if isinstance(allocation, Tensor):
            compressed = torch.bmm(allocation, projected)
            denominator = allocation.sum(dim=-1)
        else:
            indices, raw = self._three_neighbours(allocation.cumulative, n_tokens)
            compressed = self._sparse_aggregate(projected, indices, raw,
                                                allocation.denominator, n_tokens)
            denominator = allocation.denominator
            allocation = DualDynamicWeights(allocation.cumulative, denominator, n_tokens)

        return (
            compressed,
            allocation,
            joint_quota.to(hidden_pos.dtype),
            self.scores_to_quota(scores_pos, n_tokens).to(hidden_pos.dtype),
            self.scores_to_quota(scores_rc_aligned, n_tokens).flip(1).to(hidden_rc.dtype),
        )

    def score_quota(self, hidden: Tensor) -> Tensor:
        length = hidden.shape[1]
        n_tokens = self.target_tokens(length)
        scores = self.allocation_scores(hidden)
        return self.scores_to_quota(scores, n_tokens).to(hidden.dtype)


class SwiGLUExpert(nn.Module):
    """One dense expert used by the sparse MoE FFN."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoE(nn.Module):
    """Full-capacity Top-2 sparse MoE following the JanusDNA design.

    All experts are resident on the current device. Every token is dispatched
    to its two highest-probability experts, with no token dropping or capacity
    overflow. The returned auxiliary loss encourages balanced routing.
    """

    def __init__(self, hidden_size: int, intermediate_size: int,
                 num_experts: int, num_experts_per_tok: int):
        super().__init__()
        if num_experts < 2:
            raise ValueError("SparseMoE requires at least two experts")
        if not 1 <= num_experts_per_tok <= num_experts:
            raise ValueError("num_experts_per_tok must be in [1, num_experts]")
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        # Evaluation can trade extra arithmetic for a static dense dispatch
        # at short sequence lengths. This avoids dynamic-sized ``where/nonzero``
        # outputs and their GPU-to-host synchronization when the token count is
        # small; longer sequences keep the sparse path to avoid excess FLOPs.
        self.fast_eval_dispatch = True
        self.fast_train_dispatch = False
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUExpert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        router_logits = self.router(flat)
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(
            routing_probs, self.num_experts_per_tok, dim=-1
        )

        output = torch.zeros_like(flat)
        dense_dispatch = (
            (not self.training and self.fast_eval_dispatch and flat.shape[0] <= 2048)
            or (self.training and self.fast_train_dispatch and flat.shape[0] <= 8192)
        )
        if dense_dispatch:
            expert_weights = routing_weights.new_zeros((self.num_experts, flat.shape[0]))
            expert_weights.scatter_add_(0, selected_experts.transpose(0, 1),
                                        routing_weights.transpose(0, 1))
            for expert_idx, expert in enumerate(self.experts):
                expert_output = expert(flat)
                weight = expert_weights[expert_idx].to(expert_output.dtype).unsqueeze(-1)
                output.add_((expert_output * weight).to(output.dtype))
            # Router auxiliary loss is still returned unchanged. Evaluation
            # callers generally ignore it, but retaining it keeps this path
            # semantically interchangeable with sparse dispatch.
            expert_mask = F.one_hot(
                selected_experts, num_classes=self.num_experts
            ).to(routing_probs.dtype)
            tokens_per_expert = expert_mask.mean(dim=(0, 1))
            router_prob_per_expert = routing_probs.mean(dim=0)
            aux_loss = self.num_experts * torch.sum(
                tokens_per_expert * router_prob_per_expert
            )
            return output.reshape(shape), aux_loss

        for expert_idx, expert in enumerate(self.experts):
            token_idx, slot = torch.where(selected_experts == expert_idx)
            if token_idx.numel() == 0:
                continue
            expert_output = expert(flat[token_idx])
            expert_output = expert_output * routing_weights[token_idx, slot].to(expert_output.dtype).unsqueeze(-1)
            # The expert matmul may be autocast to BF16 while ``flat`` (and
            # therefore the accumulation buffer) remains FP32.  Accumulate in
            # the buffer dtype so mixed-precision training works correctly.
            output.index_add_(0, token_idx, expert_output.to(output.dtype))

        # Sparse routing means a rank may not dispatch any token to some
        # experts.  Connect every expert parameter to the graph with a zero
        # contribution so vanilla DDP sees the same parameter set on every
        # rank, without enabling the expensive find_unused_parameters scan.
        unused_expert_zero = output.new_zeros(())
        for expert in self.experts:
            for parameter in expert.parameters():
                unused_expert_zero = unused_expert_zero + parameter.float().sum() * 0.0
        output = output + unused_expert_zero.to(output.dtype)

        expert_mask = F.one_hot(
            selected_experts, num_classes=self.num_experts
        ).to(routing_probs.dtype)
        tokens_per_expert = expert_mask.mean(dim=(0, 1))
        router_prob_per_expert = routing_probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(
            tokens_per_expert * router_prob_per_expert
        )
        return output.reshape(shape), aux_loss


class RotarySelfAttentionBlock(nn.Module):
    def __init__(self, config: VeriCompConfig, *, use_moe: bool = False):
        super().__init__()
        if config.d_model % config.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.nhead = config.nhead
        self.head_dim = config.d_model // config.nhead
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.use_moe = use_moe
        if use_moe:
            self.moe = SparseMoE(
                config.d_model,
                config.moe_intermediate_size,
                config.num_experts,
                config.num_experts_per_tok,
            )
        else:
            self.ff1 = nn.Linear(config.d_model, config.dim_feedforward)
            self.ff2 = nn.Linear(config.dim_feedforward, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.force_flash_attention = config.force_flash_attention
        self.causal_attention = config.causal_attention
        inv = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def _rope(self, x: Tensor, rotary_cache: Optional[tuple[Tensor, Tensor]] = None) -> Tensor:
        # x: [B, H, N, Dh]
        if rotary_cache is None:
            positions = torch.arange(x.shape[2], device=x.device, dtype=self.inv_freq.dtype)
            angles = torch.outer(positions, self.inv_freq)
            cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        else:
            cos, sin = rotary_cache
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, x: Tensor, rotary_cache: Optional[tuple[Tensor, Tensor]] = None) -> tuple[Tensor, Tensor]:
        bsz, length, dim = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q = self._rope(q.view(bsz, length, self.nhead, self.head_dim).transpose(1, 2), rotary_cache)
        k = self._rope(k.view(bsz, length, self.nhead, self.head_dim).transpose(1, 2), rotary_cache)
        v = v.view(bsz, length, self.nhead, self.head_dim).transpose(1, 2)
        dropout_p = self.dropout.p if self.training else 0.0
        if self.force_flash_attention:
            if q.device.type != "cuda":
                raise RuntimeError(
                    "VeriComp is configured with force_flash_attention=True, "
                    "but the input is not on CUDA."
                )
            if q.dtype not in (torch.float16, torch.bfloat16):
                raise RuntimeError(
                    f"Flash Attention requires float16 or bfloat16, got {q.dtype}. "
                    "Use CUDA autocast or disable force_flash_attention."
                )
            # Restrict PyTorch SDPA to the fused Flash Attention backend. If
            # the shape, dtype, GPU, or build is unsupported, PyTorch raises
            # instead of silently falling back to math attention.
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                attn = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=dropout_p,
                    is_causal=self.causal_attention,
                )
        else:
            attn = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p,
                is_causal=self.causal_attention,
            )
        attn = attn.transpose(1, 2).contiguous().view(bsz, length, dim)
        x = x + self.dropout(self.out(attn))
        ff_input = self.norm2(x)
        if self.use_moe:
            ff_output, router_aux_loss = self.moe(ff_input)
        else:
            ff_output = self.ff2(F.gelu(self.ff1(ff_input)))
            router_aux_loss = x.new_zeros(())
        x = x + self.dropout(ff_output)
        return x, router_aux_loss


class TransformerBottleneckBackbone(nn.Module):
    def __init__(self, config: VeriCompConfig):
        super().__init__()
        self.activation_checkpointing = config.activation_checkpointing
        if config.expert_layer_period <= 0:
            raise ValueError("expert_layer_period must be positive")
        self.layers = nn.ModuleList()
        for layer_index in range(config.num_layers):
            use_moe = (
                layer_index % config.expert_layer_period == config.expert_layer_offset
            )
            layer = RotarySelfAttentionBlock(config, use_moe=use_moe)
            self.layers.append(layer)
        self.norm = nn.LayerNorm(config.d_model)
        # Every block uses the same RoPE frequencies. Compute the position
        # phases once per forward instead of rebuilding sin/cos in every
        # layer (and separately for Q and K).
        self.share_rotary_embeddings = True

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        router_aux_loss = x.new_zeros(())
        rotary_cache = None
        if self.share_rotary_embeddings and self.layers:
            inv_freq = self.layers[0].inv_freq
            positions = torch.arange(x.shape[1], device=x.device, dtype=inv_freq.dtype)
            angles = torch.outer(positions, inv_freq)
            rotary_cache = (angles.cos().to(x.dtype), angles.sin().to(x.dtype))
        for layer in self.layers:
            if self.training and self.activation_checkpointing:
                x, layer_aux_loss = activation_checkpoint(
                    layer, x, rotary_cache, use_reentrant=False
                )
            else:
                x, layer_aux_loss = layer(x, rotary_cache)
            router_aux_loss = router_aux_loss + layer_aux_loss
        return self.norm(x), router_aux_loss / max(1, sum(layer.use_moe for layer in self.layers))


class DualDynamicInverseUpsamplerAndDecoder(nn.Module):
    def __init__(self, config: VeriCompConfig):
        super().__init__()
        self.chunk_size = config.upsample_chunk_size
        # Two kernel-5 convolutions each have radius 2, so an output position
        # has a total receptive-field radius of 4.  This halo makes chunked
        # fusion identical to running the convolutions over the full length.
        self.fusion_halo = 4
        self.activation_checkpointing = config.activation_checkpointing
        # The decoder fusion now processes only the decompressed bottleneck
        # stream. The full-resolution encoder state enters through the
        # explicit residual path below instead of a feature concatenation.
        self.fuse1 = nn.Conv1d(config.d_model, config.d_model, 5, padding=2)
        self.fuse2 = nn.Conv1d(config.d_model, config.d_model, 5, padding=2)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        # Project the full-resolution shallow state into decoder width and add
        # it after fusion through a learned, initially-small gate.
        self.residual_norm = nn.LayerNorm(config.shallow_dim)
        self.residual_proj = nn.Linear(config.shallow_dim, config.d_model)
        self.residual_gate = nn.Linear(config.shallow_dim, 1)
        nn.init.constant_(self.residual_gate.bias, -2.0)
        self.head = nn.Linear(config.d_model, config.output_vocab_size)

    @staticmethod
    def _upsample_sparse_chunk(weights: DualDynamicWeights, z_out: Tensor,
                               start: int, stop: int) -> Tensor:
        c = weights.cumulative[:, start:stop]
        idx, raw = QuotaDynamicCompressor1M._three_neighbours(c, weights.n_tokens)
        bsz, length = c.shape
        dim = z_out.shape[-1]
        result = z_out.new_zeros((bsz, length, dim))
        for j in range(3):
            gathered = torch.gather(z_out, 1, idx[:, j].unsqueeze(-1).expand(-1, -1, dim))
            result.add_(gathered * (raw[:, j].to(z_out.dtype) / weights.denominator.gather(1, idx[:, j]).clamp_min(torch.finfo(weights.denominator.dtype).eps).to(z_out.dtype)).unsqueeze(-1))
        return result

    @staticmethod
    def _upsample_grid_chunk(weights: GridWeights, z_out: Tensor,
                             start: int, stop: int) -> Tensor:
        stride = weights.stride
        first = start // stride
        last = min(-(-stop // stride), z_out.shape[1])
        repeated = z_out[:, first:last].repeat_interleave(stride, dim=1)
        offset = start - first * stride
        return repeated[:, offset:offset + (stop - start)]

    def _upsample_chunk(self, z_out: Tensor, allocation: DualDynamicWeights | Tensor,
                        start: int, stop: int) -> Tensor:
        if isinstance(allocation, Tensor):
            return torch.bmm(allocation[:, :, start:stop].transpose(1, 2), z_out)
        if isinstance(allocation, GridWeights):
            return self._upsample_grid_chunk(allocation, z_out, start, stop)
        return self._upsample_sparse_chunk(allocation, z_out, start, stop)

    def forward(self, z_out: Tensor, allocation: DualDynamicWeights | Tensor,
                high: Tensor, *, return_h_up: bool = False,
                return_h_dec: bool = False) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Upsample and decode in L-sized chunks.

        Chunking is along the original base axis, not the bottleneck axis.
        The returned logits are exactly the same as the non-chunked result;
        only intermediate activation lifetime is reduced.
        """
        length = high.shape[1]
        logits_chunks = []
        h_up_chunks = [] if return_h_up else None
        h_dec_chunks = [] if return_h_dec else None
        halo = self.fusion_halo
        for start in range(0, length, self.chunk_size):
            stop = min(start + self.chunk_size, length)
            left = max(0, start - halo)
            right = min(length, stop + halo)
            h_up_local = self._upsample_chunk(z_out, allocation, left, right)
            x = h_up_local
            if self.training and self.activation_checkpointing:
                x = activation_checkpoint(self._fuse, x, use_reentrant=False)
            else:
                x = self._fuse(x)
            center_start, center_stop = start - left, stop - left
            x = x[:, center_start:center_stop]
            # Add the full-resolution residual after the compressed path has
            # been decoded.  This preserves local information discarded by
            # quota pooling without creating a second compression stage.
            residual = self._residual_skip(high[:, left:right])
            residual = residual[:, center_start:center_stop]
            x = x + residual
            logits_chunks.append(self.head(x))
            if h_up_chunks is not None:
                h_up_chunks.append(h_up_local[:, center_start:center_stop])
            if h_dec_chunks is not None:
                h_dec_chunks.append(x)
        logits = torch.cat(logits_chunks, dim=1)
        h_up = torch.cat(h_up_chunks, dim=1) if h_up_chunks is not None else None
        h_dec = torch.cat(h_dec_chunks, dim=1) if h_dec_chunks is not None else None
        return logits, h_up, h_dec

    def _fuse(self, x: Tensor) -> Tensor:
        x = self.fuse1(x.transpose(1, 2)).transpose(1, 2)
        x = F.gelu(self.norm1(x))
        x = self.fuse2(x.transpose(1, 2)).transpose(1, 2)
        return F.gelu(self.norm2(x))

    def _residual_skip(self, high: Tensor) -> Tensor:
        normalized = self.residual_norm(high)
        residual = self.residual_proj(normalized)
        gate = torch.sigmoid(self.residual_gate(high))
        return residual * gate


class VeriCompPretrainModel(nn.Module):
    def __init__(self, config: Optional[VeriCompConfig] = None):
        super().__init__()
        self.config = config or VeriCompConfig()
        self.encoder = DNAEmbeddingAndShallowEncoder(self.config)
        self.compressor = QuotaDynamicCompressor1M(self.config)
        self.bottleneck = TransformerBottleneckBackbone(self.config)
        self.decoder = DualDynamicInverseUpsamplerAndDecoder(self.config)
        # Downstream fine-tuning can use static MoE dispatch for small inputs.
        # Pretraining configurations with activation checkpointing retain the
        # existing sparse route unless the caller explicitly overrides it.
        self.short_sequence_dense_dispatch = True
        if self.config.duplex_mode not in {
            "dual", "forward", "rc_only", "dual_independent"
        }:
            raise ValueError(
                "duplex_mode must be dual, forward, rc_only, or dual_independent"
            )

    def forward(self, input_ids_pos: Tensor, input_ids_rc: Optional[Tensor] = None,
                labels: Optional[Tensor] = None, *, return_dense_w: bool = False,
                return_h_up: bool = False, return_bottleneck: bool = False,
                return_decoder_feature: bool = False,
                return_base_feature: bool = False):
        use_short_dense_dispatch = (
            self.short_sequence_dense_dispatch
            and self.training
            and not self.config.activation_checkpointing
            and input_ids_pos.device.type == "cuda"
            and input_ids_pos.shape[1] <= 2048
            and input_ids_pos.numel() <= 8192
        )
        for layer in self.bottleneck.layers:
            if layer.use_moe:
                layer.moe.fast_train_dispatch = use_short_dense_dispatch
        high_pos = self.encoder(input_ids_pos)
        b_pos_proposal = None
        b_rc_proposal = None
        loss_strand = high_pos.new_zeros(())
        b_rc = None
        if (input_ids_rc is not None
                and self.config.allocation_mode != "grid"
                and self.config.duplex_mode in {"dual", "dual_independent"}):
            high_rc = self.encoder(input_ids_rc)
            # Both strands propose compression evidence. The production Dual
            # averages raw evidence; Dual-independent normalizes each strand
            # to the full budget before averaging.
            z, allocation, b_pos, b_pos_proposal, b_rc_proposal = (
                self.compressor.forward_pair(
                    high_pos, high_rc, force_dense=return_dense_w,
                    independent_normalization=(
                        self.config.duplex_mode == "dual_independent"
                    ),
                )
            )
            b_rc = b_rc_proposal
            # This is a symmetric disagreement diagnostic between the two
            # proposals. It does not select either strand as ground truth and
            # is deliberately excluded from the autograd graph.
            with torch.no_grad():
                loss_strand = quota_js_divergence(b_pos_proposal, b_rc_proposal)
        else:
            z, allocation, b_pos = self.compressor(
                high_pos, force_dense=return_dense_w,
                quota_keep_mask=self.compressor.quota_keep_mask_for(
                    input_ids_pos, high_pos.shape[1]),
            )
        z_out, loss_router = self.bottleneck(z)
        if return_bottleneck and not return_decoder_feature and not return_base_feature:
            # Downstream sequence tasks pool the actual Transformer/MoE
            # bottleneck tokens, before any DeChunk, decoder, or residual
            # skip path. This also avoids allocating decoder activations.
            return {
                "bottleneck": z_out,
                "loss_router": loss_router,
                "b_pos": b_pos,
                "b_rc": b_rc,
            }
        logits, h_up, decoder_feature = self.decoder(
            z_out, allocation, high_pos, return_h_up=return_h_up,
            return_h_dec=return_decoder_feature or return_base_feature,
        )
        if return_base_feature:
            return {
                "base_feature": decoder_feature,
                "bottleneck": z_out,
                "loss_router": loss_router,
                "b_pos": b_pos,
                "b_rc": b_rc,
            }
        loss_mlm = logits.new_zeros(())
        if labels is not None:
            loss_mlm = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
        loss_moe = self.config.router_aux_loss_coef * loss_router.to(loss_mlm.dtype)
        # The strand metric is observation-only. The actual compression quota
        # was already decided symmetrically from both strands above.
        loss = loss_mlm + loss_moe
        dense_w = allocation if isinstance(allocation, Tensor) else allocation
        return {
            "logits": logits,
            "loss": loss,
            "loss_mlm": loss_mlm,
            "loss_strand": loss_strand,
            "loss_router": loss_router,
            "loss_moe": loss_moe,
            "W_norm": dense_w,
            "b_pos": b_pos,
            "b_rc": b_rc,
            "b_pos_proposal": b_pos_proposal,
            "b_rc_proposal": b_rc_proposal,
            "h_up": h_up,
            "decoder_feature": decoder_feature,
        }

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))
