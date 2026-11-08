"""Unified chromosome-level TISP evaluation.

The benchmark used by the active TISP runs scores the complete evaluable
chromosome rather than averaging PCCs from a small number of windows.  Each
100-kb input contributes its 50-kb scored centre, neighbouring centres overlap
by 25 kb, and the overlap is blended with a centre-weighted window.  The
resulting chromosome signal is then transformed and reduced to the five
strand-paired assays.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from finetune.TISP.external_tisp import TISP_ASSAY_GROUPS, _unknown_positions
from vericomp.data import IndexedFasta
from vericomp.tisp_data import TISPTracks, TISP_SEQUENCE_LENGTH


SCORE_LENGTH = 50_000
CONTEXT_LENGTH = TISP_SEQUENCE_LENGTH
CONTEXT_TRIM = (CONTEXT_LENGTH - SCORE_LENGTH) // 2
EVAL_STRIDE = 25_000
N_FLANK = 500


def _resolve_chrom(index: dict[str, tuple], requested: str) -> str:
    if requested in index:
        return requested
    alt = requested.removeprefix("chr") if requested.startswith("chr") else "chr" + requested
    if alt in index:
        return alt
    raise KeyError(requested)


def _ids(sequence: str) -> torch.Tensor:
    table = {"A": 0, "C": 1, "G": 2, "T": 3}
    return torch.tensor([table.get(base, 5) for base in sequence], dtype=torch.long)


class ChromosomeTISP(Dataset):
    """Full-chromosome windows with padded boundary contexts.

    Starts are spaced by 25 kb and include -25 kb and the final end-aligned
    start.  Thus every chromosome base is covered by at least one scored
    centre, while boundary contexts are explicitly padded with ``N`` and are
    excluded from the target mask by the evaluator.
    """

    def __init__(self, root: str | Path, chroms: list[str]):
        self.root = str(root)
        fasta_path = Path(root) / "tisp/seqs/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
        self.fasta = IndexedFasta(fasta_path)
        self.tracks = TISPTracks(root)
        self.windows: list[tuple[str, int]] = []
        self.chrom_lengths: dict[str, int] = {}
        seen_chroms: set[str] = set()
        for requested in chroms:
            chrom = _resolve_chrom(self.fasta.index, requested)
            if chrom in seen_chroms:
                continue
            seen_chroms.add(chrom)
            length = int(self.fasta.index[chrom][0])
            if length < SCORE_LENGTH:
                continue
            last_start = length - (CONTEXT_LENGTH - CONTEXT_TRIM)
            starts = list(range(-CONTEXT_TRIM, last_start + 1, EVAL_STRIDE))
            if not starts or starts[-1] != last_start:
                starts.append(last_start)
            self.windows.extend((chrom, int(start)) for start in starts)
            self.chrom_lengths[chrom] = length
        if not self.windows:
            raise ValueError("No chromosome windows available for TISP evaluation")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        chrom, start = self.windows[int(index)]
        chrom_len = self.chrom_lengths[chrom]
        end = start + CONTEXT_LENGTH
        left = max(0, start)
        right = min(chrom_len, end)
        left_pad = max(0, -start)
        right_pad = max(0, end - chrom_len)
        sequence = (
            "N" * left_pad
            + self.fasta.fetch(chrom, left, right)
            + "N" * right_pad
        )
        if len(sequence) != CONTEXT_LENGTH:
            raise RuntimeError(f"invalid padded sequence length for {chrom}:{start}")
        target = np.zeros((CONTEXT_LENGTH, 10), dtype=np.float32)
        if right > left:
            values = self.tracks.get(chrom, left, right)
            target[left_pad:left_pad + values.shape[0]] = values
        return {"input_ids": _ids(sequence), "target": torch.from_numpy(target)}


class _PCC:
    def __init__(self):
        self.n = 0
        self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0

    def update(self, x: np.ndarray, y: np.ndarray, valid: np.ndarray):
        mask = np.isfinite(x) & np.isfinite(y) & valid
        if not np.any(mask):
            return
        x = np.asarray(x[mask], dtype=np.float64)
        y = np.asarray(y[mask], dtype=np.float64)
        self.n += int(x.size)
        self.sx += float(x.sum()); self.sy += float(y.sum())
        self.sxx += float(np.dot(x, x)); self.syy += float(np.dot(y, y))
        self.sxy += float(np.dot(x, y))

    def value(self) -> float:
        if self.n <= 1:
            return float("nan")
        n = float(self.n)
        vx = self.sxx - self.sx * self.sx / n
        vy = self.syy - self.sy * self.sy / n
        cov = self.sxy - self.sx * self.sy / n
        return float(cov / math.sqrt(vx * vy)) if vx > 0 and vy > 0 else float("nan")


def _transform(values: np.ndarray, name: str) -> np.ndarray:
    if name == "cage_like":
        return np.log10(np.maximum(np.power(10.0, values) - 1.0 + 0.1, 1e-12))
    if name == "natlog_like":
        return values / math.log(10.0)
    raise ValueError(name)


def _update_assays(stats: dict[str, _PCC], pred: np.ndarray, target: np.ndarray, valid: np.ndarray):
    for assay, (plus, minus, transform) in TISP_ASSAY_GROUPS.items():
        pred_plus = _transform(pred[:, plus], transform)
        pred_minus = _transform(pred[:, minus], transform)
        target_plus = _transform(target[:, plus], transform)
        target_minus = _transform(target[:, minus], transform)
        stats[assay].update(np.concatenate((pred_plus, pred_minus)),
                            np.concatenate((target_plus, target_minus)),
                            np.concatenate((valid, valid)))


def _merge_n_intervals(dataset: ChromosomeTISP, chrom: str) -> list[tuple[int, int]]:
    positions = _unknown_positions(dataset, chrom, dataset.chrom_lengths[chrom])
    if positions.size == 0:
        return []
    starts = np.maximum(positions - N_FLANK, 0)
    ends = np.minimum(positions + N_FLANK + 1, dataset.chrom_lengths[chrom])
    result: list[tuple[int, int]] = []
    left, right = int(starts[0]), int(ends[0])
    for start, end in zip(starts[1:], ends[1:]):
        start, end = int(start), int(end)
        if start <= right:
            right = max(right, end)
        else:
            result.append((left, right)); left, right = start, end
    result.append((left, right))
    return result


def _valid_range(start: int, end: int, intervals: list[tuple[int, int]]) -> np.ndarray:
    valid = np.ones(end - start, dtype=bool)
    for left, right in intervals:
        if right <= start:
            continue
        if left >= end:
            break
        valid[max(left, start) - start:min(right, end) - start] = False
    return valid


def _blend_weights() -> np.ndarray:
    # Centre-weighted but non-zero at the edges, so no base is discarded when
    # a chromosome is covered by adjacent windows.
    x = (np.arange(SCORE_LENGTH, dtype=np.float32) + 0.5) / SCORE_LENGTH
    return (0.05 + 0.95 * (0.5 - 0.5 * np.cos(2.0 * np.pi * x))).astype(np.float32)


class _StitchBuffer:
    def __init__(self, chrom: str, intervals: list[tuple[int, int]], stats: dict[str, _PCC]):
        self.chrom = chrom
        self.intervals = intervals
        self.stats = stats
        self.start: int | None = None
        self.pred_sum = np.zeros((0, 10), dtype=np.float32)
        self.target_sum = np.zeros((0, 10), dtype=np.float32)
        self.weight_sum = np.zeros((0,), dtype=np.float32)
        self.weights = _blend_weights()

    def _flush(self, end: int):
        if self.start is None or end <= self.start:
            return
        count = end - self.start
        weights = self.weight_sum[:count]
        nonzero = weights > 0
        pred = self.pred_sum[:count] / np.maximum(weights[:, None], 1e-12)
        target = self.target_sum[:count] / np.maximum(weights[:, None], 1e-12)
        valid = nonzero & _valid_range(self.start, end, self.intervals)
        _update_assays(self.stats, pred, target, valid)
        self.pred_sum = self.pred_sum[count:]
        self.target_sum = self.target_sum[count:]
        self.weight_sum = self.weight_sum[count:]
        self.start = end

    def add(self, start: int, pred: np.ndarray, target: np.ndarray, local_start: int, local_end: int):
        end = start + (local_end - local_start)
        if self.start is None:
            self.start = start
        if start < self.start:
            raise ValueError("chromosome windows are not sorted")
        if start > self.start:
            self._flush(start)
        offset = start - self.start
        needed = offset + (local_end - local_start)
        if needed > len(self.weight_sum):
            extra = needed - len(self.weight_sum)
            self.pred_sum = np.concatenate((self.pred_sum, np.zeros((extra, 10), dtype=np.float32)))
            self.target_sum = np.concatenate((self.target_sum, np.zeros((extra, 10), dtype=np.float32)))
            self.weight_sum = np.concatenate((self.weight_sum, np.zeros(extra, dtype=np.float32)))
        weights = self.weights[local_start:local_end]
        sl = slice(offset, offset + len(weights))
        self.pred_sum[sl] += pred[local_start:local_end] * weights[:, None]
        self.target_sum[sl] += target[local_start:local_end] * weights[:, None]
        self.weight_sum[sl] += weights

    def finish(self):
        if self.start is not None:
            self._flush(self.start + len(self.weight_sum))


@torch.inference_mode()
def evaluate_chromosome_level(model, loader, device: torch.device, dataset: ChromosomeTISP) -> dict[str, object]:
    """Evaluate complete chromosomes with weighted overlap stitching."""
    model_was_training = model.training
    model.eval()
    interval_cache = {chrom: _merge_n_intervals(dataset, chrom) for chrom in dataset.chrom_lengths}
    stats_by_chrom = {
        chrom: {name: _PCC() for name in TISP_ASSAY_GROUPS}
        for chrom in dataset.chrom_lengths
    }
    buffers: dict[str, _StitchBuffer] = {}
    total_loss = 0.0
    batches = 0
    window_index = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        target = batch["target"].numpy()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda" and bool(getattr(model, "use_bf16", False))):
            output = model(input_ids, batch["target"].to(device, non_blocking=True))
        predictions = output["pred"].float().cpu().numpy()
        total_loss += float(output["loss"].float().item())
        batches += 1
        for row in range(predictions.shape[0]):
            chrom, window_start = dataset.windows[window_index]
            window_index += 1
            chrom_len = dataset.chrom_lengths[chrom]
            nominal_start = int(window_start) + CONTEXT_TRIM
            nominal_end = nominal_start + SCORE_LENGTH
            segment_start = max(0, nominal_start)
            segment_end = min(chrom_len, nominal_end)
            if segment_end <= segment_start:
                continue
            local_start = segment_start - nominal_start
            local_end = segment_end - nominal_start
            if chrom not in buffers:
                buffers[chrom] = _StitchBuffer(chrom, interval_cache[chrom], stats_by_chrom[chrom])
            buffers[chrom].add(segment_start, predictions[row], target[row], local_start, local_end)
    if window_index != len(dataset.windows):
        raise ValueError(f"evaluator consumed {window_index} of {len(dataset.windows)} windows")
    for buffer in buffers.values():
        buffer.finish()
    if model_was_training:
        model.train()

    per_chrom: dict[str, dict[str, float]] = {}
    for chrom, stats in stats_by_chrom.items():
        values = {f"{name}_pcc": stat.value() for name, stat in stats.items()}
        finite = [value for value in values.values() if np.isfinite(value)]
        values["avg_pcc"] = float(np.mean(finite)) if finite else float("nan")
        per_chrom[chrom] = values
    all_values = [values["avg_pcc"] for values in per_chrom.values() if np.isfinite(values["avg_pcc"])]
    return {
        "avg_pcc": float(np.mean(all_values)) if all_values else float("nan"),
        "per_chromosome": per_chrom,
        "loss": total_loss / max(1, batches),
        "windows": len(dataset),
        "metric": "chromosome-level weighted overlap stitching, five strand-paired PCCs",
        "score_length": SCORE_LENGTH,
        "eval_stride": EVAL_STRIDE,
    }
