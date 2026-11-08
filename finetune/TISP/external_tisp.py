"""HyenaDNA TISP fine-tuning on the DNALongBench protocol.

This runner keeps the TISP data, masking, transforms, scoring, and validation
selection identical to ``finetune.TISP.tisp`` while adapting the HyenaDNA
single-base hidden states to the ten-track regression head.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from vericomp.tisp_data import (
    TISP_SEQUENCE_LENGTH,
    TISPRandomDataset,
    TISPWindowDataset,
)


_BASES = ("A", "C", "G", "T", "N", "N")
TISP_WINDOW_STEP = 50_000
TISP_EVAL_TRIM_BP = (TISP_SEQUENCE_LENGTH - TISP_WINDOW_STEP) // 2
TISP_VALID_MASK_FLANK_BP = 500
TISP_ASSAY_GROUPS = {
    "fc": (0, 9, "cage_like"),
    "ec": (1, 8, "cage_like"),
    "er": (2, 7, "cage_like"),
    "gc": (3, 6, "natlog_like"),
    "pc": (4, 5, "natlog_like"),
}


class _TorchPearsonAccumulator:
    """Streaming Pearson statistics kept on the inference device.

    The previous implementation copied every 50 kb segment to CPU and ran a
    NumPy dot product for each assay.  Keeping the six sufficient statistics
    on GPU removes that per-window synchronization while preserving the same
    chromosome-level aggregation formula.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.count = torch.zeros((), dtype=torch.float64, device=device)
        self.sum_x = torch.zeros((), dtype=torch.float64, device=device)
        self.sum_y = torch.zeros((), dtype=torch.float64, device=device)
        self.sum_x2 = torch.zeros((), dtype=torch.float64, device=device)
        self.sum_y2 = torch.zeros((), dtype=torch.float64, device=device)
        self.sum_xy = torch.zeros((), dtype=torch.float64, device=device)

    def update(self, x: torch.Tensor, y: torch.Tensor, valid_mask: torch.Tensor | None = None):
        x = x.float()
        y = y.float()
        valid = torch.isfinite(x) & torch.isfinite(y)
        if valid_mask is not None:
            valid &= valid_mask.to(device=x.device, dtype=torch.bool)
        if not bool(valid.any()):
            return
        x = x[valid].to(dtype=torch.float64)
        y = y[valid].to(dtype=torch.float64)
        self.count += x.numel()
        self.sum_x += x.sum()
        self.sum_y += y.sum()
        self.sum_x2 += torch.square(x).sum()
        self.sum_y2 += torch.square(y).sum()
        self.sum_xy += (x * y).sum()

    def compute(self) -> float:
        count = float(self.count.item())
        if count <= 1:
            return float("nan")
        sum_x = float(self.sum_x.item())
        sum_y = float(self.sum_y.item())
        var_x = float(self.sum_x2.item()) - sum_x * sum_x / count
        var_y = float(self.sum_y2.item()) - sum_y * sum_y / count
        cov = float(self.sum_xy.item()) - sum_x * sum_y / count
        return float(cov / math.sqrt(var_x * var_y)) if var_x > 0 and var_y > 0 else float("nan")


def _transform_tisp_values(values: torch.Tensor, transform_name: str) -> torch.Tensor:
    if transform_name == "cage_like":
        return torch.log10(torch.pow(10.0, values) - 1.0 + 0.1)
    if transform_name == "natlog_like":
        return values / math.log(10.0)
    raise ValueError(f"unknown TISP transform {transform_name!r}")


def _update_assay_accumulators(
    accumulators: dict[str, _TorchPearsonAccumulator],
    predictions: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> None:
    for assay_name, (plus_idx, minus_idx, transform_name) in TISP_ASSAY_GROUPS.items():
        pred_plus = _transform_tisp_values(predictions[:, :, plus_idx], transform_name)
        pred_minus = _transform_tisp_values(predictions[:, :, minus_idx], transform_name)
        label_plus = _transform_tisp_values(labels[:, :, plus_idx], transform_name)
        label_minus = _transform_tisp_values(labels[:, :, minus_idx], transform_name)
        mask = None if valid_mask is None else valid_mask.to(
            device=pred_plus.device, dtype=torch.bool
        )
        accumulator = accumulators[assay_name]
        accumulator.update(pred_plus, label_plus, mask)
        accumulator.update(pred_minus, label_minus, mask)


def _unknown_positions(dataset: TISPWindowDataset, chrom: str, chrom_len: int) -> np.ndarray:
    sequence = dataset.fasta.fetch(chrom, 0, chrom_len)
    if len(sequence) != int(chrom_len):
        raise ValueError(
            f"TISP FASTA returned {len(sequence)} bases for {chrom}, expected {chrom_len}."
        )
    sequence_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    return np.flatnonzero(
        (sequence_bytes == ord("N")) | (sequence_bytes == ord("n"))
    ).astype(np.int64, copy=False)


def _segment_valid_mask(
    n_positions: np.ndarray, segment_start: int, segment_end: int
) -> torch.Tensor:
    valid_mask = np.ones(segment_end - segment_start, dtype=np.bool_)
    if n_positions.size == 0:
        return torch.from_numpy(valid_mask)
    left = int(np.searchsorted(
        n_positions, segment_start - TISP_VALID_MASK_FLANK_BP, side="left"
    ))
    right = int(np.searchsorted(
        n_positions, segment_end + TISP_VALID_MASK_FLANK_BP, side="left"
    ))
    for n_position in n_positions[left:right]:
        invalid_start = max(segment_start, int(n_position) - TISP_VALID_MASK_FLANK_BP)
        invalid_end = min(segment_end, int(n_position) + TISP_VALID_MASK_FLANK_BP + 1)
        valid_mask[
            invalid_start - segment_start : invalid_end - segment_start
        ] = False
    return torch.from_numpy(valid_mask)


def _flush_segment(
    accumulators: dict[str, _TorchPearsonAccumulator],
    predictions: torch.Tensor,
    labels: torch.Tensor,
    segment_start: int,
    segment_end: int,
    n_positions: np.ndarray,
) -> None:
    if segment_end <= segment_start:
        return
    valid_mask = _segment_valid_mask(n_positions, segment_start, segment_end)
    if bool(valid_mask.any()):
        valid_mask = valid_mask.to(device=predictions.device)
        _update_assay_accumulators(
            accumulators,
            predictions.unsqueeze(0),
            labels.unsqueeze(0),
            valid_mask=valid_mask.unsqueeze(0),
        )


def evaluate_tisp_chromosome_reference(
    model: nn.Module, loader: DataLoader, device: torch.device, *,
    use_bf16_autocast: bool = True,
) -> dict[str, float]:
    """GeneZip-compatible chromosome-level TISP evaluator.

    Center segments are stitched in genomic order. Each overlapping base is
    counted once, and unknown bases plus a 500-bp flank are masked using the
    full chromosome sequence rather than the individual input window.
    """
    dataset = loader.dataset
    if not isinstance(dataset, TISPWindowDataset):
        raise TypeError("chromosome-level TISP evaluation requires TISPWindowDataset")

    model_was_training = model.training
    model.eval()
    accumulators = {
        name: _TorchPearsonAccumulator(device) for name in TISP_ASSAY_GROUPS
    }
    n_cache: dict[str, np.ndarray] = {}
    windows = list(dataset.windows)
    window_index = 0
    current_chrom = None
    pending = None
    losses = []

    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and use_bf16_autocast,
            ):
                output = model(input_ids, target)
            predictions = output["pred"].float().detach()
            labels = target.float().detach()
            losses.append(float(output["loss"].float().item()))

            for batch_idx in range(predictions.shape[0]):
                if window_index >= len(windows):
                    raise ValueError("TISP evaluator consumed more windows than dataset")
                chrom, window_start = windows[window_index]
                window_index += 1
                chrom_len = int(dataset.fasta.index[chrom][0])

                if chrom != current_chrom:
                    if pending is not None:
                        _flush_segment(
                            accumulators, pending["predictions"], pending["labels"],
                            pending["segment_start"], pending["segment_end"],
                            n_cache[current_chrom],
                        )
                        pending = None
                    current_chrom = chrom
                    if chrom not in n_cache:
                        n_cache[chrom] = _unknown_positions(dataset, chrom, chrom_len)

                segment_start = int(window_start) + TISP_EVAL_TRIM_BP
                segment_end = int(window_start) + TISP_SEQUENCE_LENGTH - TISP_EVAL_TRIM_BP
                center = slice(TISP_EVAL_TRIM_BP, TISP_SEQUENCE_LENGTH - TISP_EVAL_TRIM_BP)
                current = {
                    "predictions": predictions[batch_idx, center, :],
                    "labels": labels[batch_idx, center, :],
                    "segment_start": segment_start,
                    "segment_end": segment_end,
                }

                if pending is not None:
                    pending_start = int(pending["segment_start"])
                    pending_end = int(pending["segment_end"])
                    if segment_start > pending_end:
                        raise ValueError(
                            f"TISP evaluator found a gap on {chrom}: "
                            f"{pending_start}:{pending_end} then {segment_start}:{segment_end}"
                        )
                    flush_end = min(pending_end, segment_start)
                    if flush_end > pending_start:
                        flush_len = flush_end - pending_start
                        _flush_segment(
                            accumulators, pending["predictions"][:flush_len],
                            pending["labels"][:flush_len], pending_start, flush_end,
                            n_cache[chrom],
                        )
                pending = current

    if window_index != len(windows):
        raise ValueError(
            f"TISP evaluator consumed {window_index} of {len(windows)} windows"
        )
    if pending is not None:
        _flush_segment(
            accumulators, pending["predictions"], pending["labels"],
            int(pending["segment_start"]), int(pending["segment_end"]),
            n_cache[current_chrom],
        )

    if model_was_training:
        model.train()
    metrics = {
        f"{name}_pcc": accumulator.compute()
        for name, accumulator in accumulators.items()
    }
    finite = [value for value in metrics.values() if np.isfinite(value)]
    metrics["avg_pcc"] = float(np.mean(finite)) if finite else float("nan")
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    metrics["examples"] = len(windows)
    return metrics


class _BatchedTorchPearsonAccumulator(_TorchPearsonAccumulator):
    """Pearson accumulator which avoids boolean indexing and synchronisation.

    The reference evaluator updates one plus/minus track at a time and asks
    the host whether each 50-kb segment is non-empty.  This implementation
    keeps the same six sufficient statistics, but updates them for a whole
    inference batch using masked arithmetic.  The only device-to-host copy is
    performed once, when the final PCC is requested.
    """

    def update(self, x: torch.Tensor, y: torch.Tensor, valid_mask: torch.Tensor | None = None):
        x64 = x.to(dtype=torch.float64)
        y64 = y.to(dtype=torch.float64)
        valid = torch.isfinite(x64) & torch.isfinite(y64)
        if valid_mask is not None:
            valid &= valid_mask.to(device=x.device, dtype=torch.bool)
        valid_f = valid.to(dtype=torch.float64)
        zero = torch.zeros((), dtype=torch.float64, device=x.device)
        x64 = torch.where(valid, x64, zero)
        y64 = torch.where(valid, y64, zero)
        self.count += valid_f.sum(dtype=torch.float64)
        self.sum_x += x64.sum(dtype=torch.float64)
        self.sum_y += y64.sum(dtype=torch.float64)
        self.sum_x2 += torch.square(x64).sum(dtype=torch.float64)
        self.sum_y2 += torch.square(y64).sum(dtype=torch.float64)
        self.sum_xy += (x64 * y64).sum(dtype=torch.float64)


def _merge_unknown_intervals(n_positions: np.ndarray, chrom_len: int) -> list[tuple[int, int]]:
    """Return merged half-open N/flank intervals for one chromosome."""
    if n_positions.size == 0:
        return []
    starts = np.maximum(n_positions - TISP_VALID_MASK_FLANK_BP, 0)
    ends = np.minimum(n_positions + TISP_VALID_MASK_FLANK_BP + 1, chrom_len)
    intervals: list[tuple[int, int]] = []
    start, end = int(starts[0]), int(ends[0])
    for next_start, next_end in zip(starts[1:], ends[1:]):
        next_start, next_end = int(next_start), int(next_end)
        if next_start <= end:
            end = max(end, next_end)
        else:
            intervals.append((start, end))
            start, end = next_start, next_end
    intervals.append((start, end))
    return intervals


def _segment_keep_ends(windows: list[tuple[str, int]]) -> list[int] | None:
    """Compute the end of each segment's unique contribution.

    For sorted windows, GeneZip flushes a pending segment at the next
    segment's start and lets the last segment cover its full center.  Thus an
    overlapping tail window is represented by a shorter keep interval on the
    preceding row.  ``None`` is reserved for unusual non-monotone inputs,
    which are delegated to the exact reference implementation.
    """
    keep_ends: list[int] = []
    for index, (chrom, window_start) in enumerate(windows):
        start = int(window_start) + TISP_EVAL_TRIM_BP
        end = start + TISP_WINDOW_STEP
        keep_end = end
        if index + 1 < len(windows):
            next_chrom, next_window_start = windows[index + 1]
            next_start = int(next_window_start) + TISP_EVAL_TRIM_BP
            if next_chrom == chrom:
                if next_start <= start:
                    return None
                if next_start > end:
                    raise ValueError(
                        f"TISP evaluator found a gap on {chrom}: {start}:{end} then "
                        f"{next_start}:{next_start + TISP_WINDOW_STEP}"
                    )
                keep_end = next_start
        keep_ends.append(keep_end)
    return keep_ends


def _update_batched_assay_accumulators(
    accumulators: dict[str, _BatchedTorchPearsonAccumulator],
    predictions: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    """Update each assay once per inference batch (plus and minus together)."""
    for assay_name, (plus_idx, minus_idx, transform_name) in TISP_ASSAY_GROUPS.items():
        pred = torch.cat(
            (
                _transform_tisp_values(predictions[:, :, plus_idx], transform_name),
                _transform_tisp_values(predictions[:, :, minus_idx], transform_name),
            ),
            dim=0,
        )
        target = torch.cat(
            (
                _transform_tisp_values(labels[:, :, plus_idx], transform_name),
                _transform_tisp_values(labels[:, :, minus_idx], transform_name),
            ),
            dim=0,
        )
        mask = torch.cat((valid_mask, valid_mask), dim=0)
        accumulators[assay_name].update(pred, target, mask)


def evaluate_tisp_chromosome(
    model: nn.Module, loader: DataLoader, device: torch.device, *,
    use_bf16_autocast: bool = True,
) -> dict[str, float]:
    """Fast chromosome-level evaluator with the same stitching semantics.

    The DNALongBench protocol uses adjacent 50-kb center segments.  For this
    common case, N/flank masks are assembled in CPU batches and all assay
    statistics are reduced once per inference batch.  If a caller supplies
    overlapping windows, the reference evaluator is used automatically, so
    its last-window-wins de-duplication behaviour remains unchanged.
    """
    dataset = loader.dataset
    if not isinstance(dataset, TISPWindowDataset):
        raise TypeError("chromosome-level TISP evaluation requires TISPWindowDataset")
    windows = list(dataset.windows)
    keep_ends = getattr(dataset, "_tisp_eval_keep_ends", None)
    if keep_ends is None or len(keep_ends) != len(windows):
        keep_ends = _segment_keep_ends(windows)
        setattr(dataset, "_tisp_eval_keep_ends", keep_ends)
    if keep_ends is None:
        return evaluate_tisp_chromosome_reference(
            model, loader, device, use_bf16_autocast=use_bf16_autocast
        )

    model_was_training = model.training
    model.eval()
    accumulators = {
        name: _BatchedTorchPearsonAccumulator(device) for name in TISP_ASSAY_GROUPS
    }
    interval_cache = getattr(dataset, "_tisp_eval_interval_cache", None)
    if interval_cache is None:
        interval_cache = {}
        for chrom in dict.fromkeys(chrom for chrom, _ in windows):
            chrom_len = int(dataset.fasta.index[chrom][0])
            interval_cache[chrom] = _merge_unknown_intervals(
                _unknown_positions(dataset, chrom, chrom_len), chrom_len
            )
        setattr(dataset, "_tisp_eval_interval_cache", interval_cache)

    losses = torch.zeros((), dtype=torch.float64, device=device)
    batch_count = 0
    window_index = 0
    center = slice(TISP_EVAL_TRIM_BP, TISP_SEQUENCE_LENGTH - TISP_EVAL_TRIM_BP)
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            batch_size = int(input_ids.shape[0])
            batch_windows = windows[window_index : window_index + batch_size]
            if len(batch_windows) != batch_size:
                raise ValueError("TISP evaluator consumed more windows than dataset")
            valid_np = np.ones((batch_size, TISP_WINDOW_STEP), dtype=np.bool_)
            for row, (chrom, window_start) in enumerate(batch_windows):
                segment_start = int(window_start) + TISP_EVAL_TRIM_BP
                segment_end = segment_start + TISP_WINDOW_STEP
                keep_end = keep_ends[window_index + row]
                if keep_end < segment_end:
                    valid_np[row, keep_end - segment_start :] = False
                for invalid_start, invalid_end in interval_cache[chrom]:
                    left = max(segment_start, invalid_start)
                    right = min(segment_end, invalid_end)
                    if right > left:
                        valid_np[row, left - segment_start : right - segment_start] = False
            valid_mask = torch.from_numpy(valid_np).to(device=device, non_blocking=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=device.type == "cuda" and use_bf16_autocast,
            ):
                output = model(input_ids, target)
            predictions = output["pred"]
            labels = target
            _update_batched_assay_accumulators(
                accumulators, predictions[:, center, :], labels[:, center, :], valid_mask
            )
            losses += output["loss"].to(dtype=torch.float64)
            batch_count += 1
            window_index += batch_size

    if window_index != len(windows):
        raise ValueError(f"TISP evaluator consumed {window_index} of {len(windows)} windows")
    if model_was_training:
        model.train()
    metrics = {f"{name}_pcc": accumulator.compute() for name, accumulator in accumulators.items()}
    finite = [value for value in metrics.values() if np.isfinite(value)]
    metrics["avg_pcc"] = float(np.mean(finite)) if finite else float("nan")
    metrics["loss"] = float((losses / max(batch_count, 1)).item())
    metrics["examples"] = len(windows)
    return metrics


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sequences(input_ids: torch.Tensor) -> list[str]:
    return ["".join(_BASES[int(base)] for base in row) for row in input_ids.cpu().tolist()]


class HyenaTISPModel(nn.Module):
    def __init__(self, model_path: str, device: torch.device):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        self.backbone.to(device)
        # TISP uses the per-base hidden state, not the next-token vocabulary
        # logits.  Keeping lm_head frozen avoids allocating its unused grads.
        for parameter in self.backbone.lm_head.parameters():
            parameter.requires_grad_(False)
        width = int(self.backbone.config.d_model)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 10)
        self.norm.to(device)
        self.head.to(device)
        self.device = device

    def forward(self, input_ids: torch.Tensor, target: torch.Tensor | None = None):
        sequences = _sequences(input_ids)
        encoded = self.tokenizer(
            sequences, return_tensors="pt", padding=False, truncation=False
        )["input_ids"].to(self.device)
        # HyenaDNA appends one [SEP] token.  Remove it to preserve the exact
        # 100,000-base alignment with the TISP target tracks.
        hidden = self.backbone.hyena(
            input_ids=encoded, output_hidden_states=False, return_dict=True
        ).last_hidden_state[:, :TISP_SEQUENCE_LENGTH]
        log_rates = self.head(self.norm(hidden))
        pred = torch.exp(log_rates.float())
        result = {"log_rates": log_rates, "pred": pred,
                  "router": log_rates.new_zeros(())}
        if target is not None:
            valid = torch.isfinite(target) & torch.isfinite(log_rates)
            if bool(valid.any()):
                rates = torch.exp(log_rates[valid].float())
                y = target[valid].float()
                task_loss = (y * torch.log((y + 1e-5) / (rates + 1e-5)) + rates - y).mean()
            else:
                task_loss = log_rates.sum() * 0.0
            result["loss_task"] = task_loss
            result["loss_router_term"] = task_loss * 0.0
            result["loss"] = task_loss
        return result


class NTv3TISPModel(nn.Module):
    """NTv3 adapter using NTv3's true final hidden state.

    The final state must be read from ``outputs["hidden_states"][-1]``.
    Hooking the last deconvolution block captures its output before NTv3 adds
    the corresponding skip connection, which is not the representation used
    by the native model.
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        *,
        all_bf16: bool = False,
        full_fp32: bool = False,
        strict_dnalongbench: bool = False,
        bf16_weights: bool = False,
    ):
        super().__init__()
        source = "/gpfs_data/Projects/GenomeModelProject/sources/ntv3_base_model"
        sys.path.insert(0, source)
        from configuration_ntv3_pretrained import Ntv3PreTrainedConfig
        from modeling_ntv3_pretrained import NTv3PreTrained
        from tokenization_ntv3 import NTv3Tokenizer

        self.tokenizer = NTv3Tokenizer.from_pretrained(model_path, local_files_only=True)
        config = Ntv3PreTrainedConfig.from_pretrained(model_path, local_files_only=True)
        # The default path follows the NTv3 mixed-precision policy.  The
        # optional all_bf16 path is used for a controlled comparison in which
        # parameters and compute, including sensitive modules, are BF16.
        mixed_precision = {
            "stem_compute_dtype": "bfloat16",
            "down_convolution_compute_dtype": "bfloat16",
            "transformer_qkvo_compute_dtype": "bfloat16",
            "transformer_ffn_compute_dtype": "bfloat16",
            "up_convolution_compute_dtype": "bfloat16",
            "modulation_compute_dtype": "bfloat16",
            "embedding_compute_dtype": "float32",
            "layernorm_compute_dtype": "float32",
            "lmhead_compute_dtype": "float32",
            "stem_param_dtype": "float32",
            "down_convolution_param_dtype": "float32",
            "transformer_qkvo_param_dtype": "float32",
            "transformer_ffn_param_dtype": "float32",
            "up_convolution_param_dtype": "float32",
            "modulation_param_dtype": "float32",
            "embedding_param_dtype": "float32",
            "layernorm_param_dtype": "float32",
            "lmhead_param_dtype": "float32",
        }
        if full_fp32 or strict_dnalongbench:
            # The published DNALongBench Lightning configurations use
            # ``precision: 32``.  NTv3's released checkpoint is also FP32, so
            # do not silently replace a requested FP32 run with the faster
            # mixed-precision path used by the historical VeriComp runs.
            mixed_precision = {key: "float32" for key in mixed_precision}
        if all_bf16:
            mixed_precision = {key: "bfloat16" for key in mixed_precision}
        for key, value in mixed_precision.items():
            setattr(config, key, value)
        self.backbone = NTv3PreTrained.from_pretrained(
            model_path, config=config, local_files_only=True
        ).to(device=device, dtype=torch.bfloat16 if all_bf16 else None)
        self.precision_policy = mixed_precision
        self.all_bf16 = all_bf16
        self.device = device
        width = int(config.conv_init_embed_dim)
        adapter_dtype = torch.bfloat16 if all_bf16 else torch.float32
        # DNALongBench's TISP decoder is a single linear projection from the
        # backbone token representation.  The historical VeriComp runner
        # used an additional LayerNorm for its own ablations; keep that path
        # intact, but allow the official protocol to disable it explicitly.
        self.norm = (
            nn.Identity()
            if strict_dnalongbench
            else nn.LayerNorm(width, dtype=adapter_dtype)
        ).to(device)
        self.head = nn.Linear(width, 10, dtype=adapter_dtype).to(device)
        if bf16_weights and not all_bf16:
            # BF16 backbone weights with an FP32 adapter; LayerNorm stays
            # FP32 because normalization statistics are precision-sensitive.
            self.backbone.to(torch.bfloat16)
            for module in self.backbone.modules():
                if isinstance(module, nn.LayerNorm):
                    module.to(torch.float32)
            self.bf16_weights = True
        else:
            self.bf16_weights = False

    def forward(self, input_ids: torch.Tensor, target: torch.Tensor | None = None):
        sequences = _sequences(input_ids)
        # NTv3 downsamples seven times.  Pad to a multiple of 2**7 so all
        # residual skip connections have identical lengths, then crop back to
        # the original 100 kb window after the final deconvolution.
        padded_length = ((TISP_SEQUENCE_LENGTH + 127) // 128) * 128
        encoded = self.tokenizer(
            [sequence + "N" * (padded_length - len(sequence)) for sequence in sequences],
            return_tensors="pt", padding=False, truncation=False,
        )["input_ids"].to(self.device)
        outputs = self.backbone(
            input_ids=encoded,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs["hidden_states"]
        if not hidden_states:
            raise RuntimeError("NTv3 did not return hidden states")
        # NTv3 appends the state after the residual skip connection.  This is
        # the representation aligned with the model's native final embedding.
        hidden = hidden_states[-1][:, :TISP_SEQUENCE_LENGTH]
        if self.all_bf16:
            log_rates = self.head(self.norm(hidden.to(torch.bfloat16)))
        else:
            # Keep the adapter FP32 under the documented mixed-precision path.
            with torch.autocast(device_type=self.device.type, enabled=False):
                log_rates = self.head(self.norm(hidden.float()))
        pred = torch.exp(log_rates)
        result = {"log_rates": log_rates, "pred": pred,
                  "router": log_rates.new_zeros(())}
        if target is not None:
            valid = torch.isfinite(target) & torch.isfinite(log_rates)
            if bool(valid.any()):
                loss_dtype = log_rates.dtype if self.all_bf16 else torch.float32
                rates = torch.exp(log_rates[valid].to(loss_dtype))
                y = target[valid].to(loss_dtype)
                task_loss = (y * torch.log((y + 1e-5) / (rates + 1e-5)) + rates - y).mean()
            else:
                task_loss = log_rates.sum() * 0.0
            result["loss_task"] = task_loss
            result["loss_router_term"] = task_loss * 0.0
            result["loss"] = task_loss
        return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("hyenadna", "ntv3"), default="hyenadna")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--grad-accumulation", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=25000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--bf16-weights", action="store_true",
                        help="store backbone weights in BF16 while the adapter and loss stay FP32")
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=("warmup", "linear"), default="warmup")
    parser.add_argument("--scheduler-start-factor", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ntv3-all-bf16", action="store_true",
                        help="use BF16 for all NTv3 parameters, sensitive modules, adapter, and loss")
    return parser.parse_args()


def main() -> None:
    from finetune.TISP.chromosome_eval import ChromosomeTISP, EVAL_STRIDE, evaluate_chromosome_level

    args = parse_args()
    _seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("HyenaDNA TISP requires CUDA")
    train_ds = TISPRandomDataset(args.data_root, 0)
    valid_ds = ChromosomeTISP(args.data_root, ["chr10", "10"])
    test_ds = ChromosomeTISP(args.data_root, ["chr8", "8", "chr9", "9"])
    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, num_workers=0),
        "valid": DataLoader(valid_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0),
    }
    if args.model == "ntv3":
        model = NTv3TISPModel(args.model_path, device, all_bf16=args.ntv3_all_bf16,
                              bf16_weights=args.bf16_weights)
    else:
        model = HyenaTISPModel(args.model_path, device)
    model.use_bf16 = device.type == "cuda" and (
        args.model == "hyenadna" or args.ntv3_all_bf16
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    warmup_steps = int(args.train_steps * args.warmup_ratio)
    if args.scheduler == "linear":
        def lr_scale(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return args.scheduler_start_factor + (1.0 - args.scheduler_start_factor) * (step + 1) / warmup_steps
            decay_steps = max(1, args.train_steps - warmup_steps)
            return max(0.0, 1.0 - (step - warmup_steps) / decay_steps)
    elif warmup_steps:
        def lr_scale(step: int) -> float:
            if step < warmup_steps:
                return args.scheduler_start_factor + (1.0 - args.scheduler_start_factor) * (step + 1) / warmup_steps
            return 1.0
    else:
        def lr_scale(step: int) -> float:
            return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps({
            "task": "TISP", "subtask": "TISP", "model": args.model,
            "model_path": str(Path(args.model_path).resolve()),
            "data_root": str(Path(args.data_root).resolve()),
            "sequence_length": TISP_SEQUENCE_LENGTH, "window_step": EVAL_STRIDE,
            "train_examples": len(train_ds), "valid_examples": len(valid_ds),
            "test_examples": len(test_ds), "arguments": vars(args),
            "effective_batch_size": args.batch_size * args.grad_accumulation,
            "dtype": (
                "bfloat16_all_modules_and_loss"
                if args.model == "ntv3" and args.ntv3_all_bf16
                else "bf16_body_fp32_embedding_layernorm_lmhead_loss"
                if args.model == "ntv3" else "bfloat16"
            ),
            "evaluator": "chromosome_weighted_overlap_stitching_paired_assay_pcc",
            "validation_best": "avg_pcc",
        }, indent=2) + "\n"
    )
    resolved = {
        "event": "resolved_config_before_training", "task": "TISP", "subtask": "TISP",
        "model": args.model, "model_path": str(Path(args.model_path).resolve()),
        "data": {"train_examples": len(train_ds), "valid_examples": len(valid_ds),
                 "test_examples": len(test_ds), "sequence_length": TISP_SEQUENCE_LENGTH,
                 "window_step": EVAL_STRIDE},
        "optimization": {"batch_size": args.batch_size, "grad_accumulation": args.grad_accumulation,
                         "effective_batch_size": args.batch_size * args.grad_accumulation,
                         "train_steps": args.train_steps, "eval_every": args.eval_every,
                         "learning_rate": args.lr, "weight_decay": args.weight_decay},
        "parameters": {"total": sum(p.numel() for p in model.parameters()),
                       "trainable": sum(p.numel() for p in params)},
        "runtime": {"device": str(device),
                    "autocast": (
                        "ntv3_internal_all_bf16"
                        if args.model == "ntv3" and args.ntv3_all_bf16
                        else "ntv3_internal_bf16_body_fp32_sensitive"
                        if args.model == "ntv3" else "bfloat16"
                    ),
                    "evaluator": "chromosome_weighted_overlap_stitching_paired_assay_pcc"},
    }
    print(json.dumps(resolved, indent=2), flush=True)

    history = []
    best_pcc = -math.inf
    iterator = iter(loaders["train"])
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    # ``train_steps`` and ``eval_every`` are optimizer-step counts, matching
    # the VeriComp TISP runner.  Microsteps are tracked separately because
    # HyenaDNA needs gradient accumulation to fit 100 kb on one card.
    optimizer_step = 0
    micro_step = 0
    while optimizer_step < args.train_steps:
        micro_step += 1
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loaders["train"])
            batch = next(iterator)
        model.train()
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(input_ids, target)
            loss = output["loss"] / args.grad_accumulation
        loss.backward()
        updated = False
        if micro_step % args.grad_accumulation == 0 or optimizer_step + 1 == args.train_steps:
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            updated = True
        if updated and (optimizer_step == 1 or optimizer_step % 50 == 0 or optimizer_step == args.train_steps):
            print(json.dumps({"step": optimizer_step, "micro_step": micro_step,
                              "train_loss": float(output["loss"].detach().float()),
                              "step_s": (time.perf_counter() - started) / optimizer_step}), flush=True)
        if updated and (optimizer_step % args.eval_every == 0 or optimizer_step == args.train_steps):
            valid = evaluate_chromosome_level(model, loaders["valid"], device, valid_ds)
            record = {"step": optimizer_step, "micro_step": micro_step,
                      "valid": valid}
            history.append(record)
            print(json.dumps(record), flush=True)
            if np.isfinite(valid["avg_pcc"]) and valid["avg_pcc"] > best_pcc:
                best_pcc = valid["avg_pcc"]
                torch.save({"model": model.state_dict(), "step": optimizer_step,
                            "valid": valid, "args": vars(args)}, output_dir / "best.pt")

    final_test = evaluate_chromosome_level(model, loaders["test"], device, test_ds)
    torch.save({"model": model.state_dict(), "step": args.train_steps,
                "test_final": final_test, "args": vars(args)}, output_dir / "last.pt")
    best_test = None
    best_checkpoint = output_dir / "best.pt"
    if best_checkpoint.exists():
        payload = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(payload["model"])
        best_test = evaluate_chromosome_level(model, loaders["test"], device, test_ds)
    result = {"task": "TISP", "model": args.model, "valid_best_avg_pcc": best_pcc,
              "test_final": final_test, "test_validation_best": best_test,
              "history": history}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "valid_best_avg_pcc": best_pcc}), flush=True)


if __name__ == "__main__":
    main()
