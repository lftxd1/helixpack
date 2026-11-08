"""NTv3 TISP fine-tuning using the published DNALongBench protocol.

This entry point is deliberately separate from the historical VeriComp TISP
runner.  It follows the public DNALongBench implementation for the parts that
are independent of the backbone: 100-kb randomly sampled training windows,
chr10 validation, chr8/chr9 sliding-window evaluation, a 10-channel linear
decoder, pseudo-Poisson KL, FP32, AdamW, cosine warmup, and an effective batch
of 16.  NTv3 was not evaluated in the DNALongBench paper, so its backbone
implementation and the model-specific learning rate are recorded explicitly
in ``run_config.json`` rather than being presented as an original result.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

from finetune.TISP.external_tisp import NTv3TISPModel, TISP_ASSAY_GROUPS
from finetune.TISP.external_tisp import evaluate_tisp_chromosome
from finetune.TISP.chromosome_eval import (
    ChromosomeTISP,
    EVAL_STRIDE,
    evaluate_chromosome_level,
)
from vericomp.data import IndexedFasta
from vericomp.tisp_data import (
    TISPTracks,
    TISP_SEQUENCE_LENGTH,
    TISP_WINDOW_STEP,
    TISPWindowDataset,
    TISPRandomDataset,
)


TRAIN_SEED = 3
GLOBAL_BATCH = 16
# Full-chromosome validation is intentionally less frequent than the old
# ten-batch window metric.  It is the metric used for validation-best.
VALIDATION_INTERVAL = 1000
VALIDATION_BATCHES = 0
WARMUP_LR_INIT = 1e-6
WEIGHT_DECAY = 0.1
ADAM_EPS = 1e-4
TRIM = (TISP_SEQUENCE_LENGTH - TISP_WINDOW_STEP) // 2


def _resolve_chrom(index: dict[str, tuple], chrom: str) -> str:
    if chrom in index:
        return chrom
    alt = chrom.removeprefix("chr") if chrom.startswith("chr") else "chr" + chrom
    if alt in index:
        return alt
    raise KeyError(chrom)


class _Intervals:
    def __init__(self, path: Path | None):
        self.by_chrom: dict[str, list[tuple[int, int]]] = {}
        if path is None or not path.exists():
            return
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip() or line.startswith(("#", "track", "browser")):
                    continue
                fields = line.split()
                if len(fields) < 3:
                    continue
                try:
                    start, end = int(fields[1]), int(fields[2])
                except ValueError:
                    continue
                if end > start:
                    self.by_chrom.setdefault(fields[0], []).append((start, end))
        for values in self.by_chrom.values():
            values.sort()

    def overlaps(self, chrom: str, start: int, end: int) -> bool:
        values = self.by_chrom.get(chrom)
        if values is None:
            alias = chrom if chrom.startswith("chr") else "chr" + chrom
            values = self.by_chrom.get(alias, ())
        return any(left < end and right > start for left, right in values)


def _ids(sequence: str) -> torch.Tensor:
    table = {"A": 0, "C": 1, "G": 2, "T": 3}
    return torch.tensor([table.get(base, 5) for base in sequence], dtype=torch.long)


class OriginalRandomTISP(IterableDataset):
    """RandomPositionsSampler-equivalent stream used by DNALongBench.

    Selene's sampler draws chromosome and centre coordinates with NumPy's
    global RNG (seed 3 in the public data loader), uses a 100-kb exclusion
    margin at both chromosome ends, and retries samples with too many unknown
    bases.  This implementation keeps the same distribution while avoiding a
    dependency on Selene in the VeriComp environment.
    """

    def __init__(self, root: str):
        self.root = Path(root)
        fasta_path = self.root / "tisp/seqs/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
        self.fasta = IndexedFasta(fasta_path)
        self.tracks = TISPTracks(root)
        self.blacklist = _Intervals(self.root / "tisp/seqs/hg38.blacklist.bed.gz")
        self.chroms = [
            c for c in self.fasta.index
            if (c in self.tracks.chrom_sizes or
                c.removeprefix("chr") in self.tracks.chrom_sizes or
                "chr" + c in self.tracks.chrom_sizes)
            if c.removeprefix("chr") not in {"8", "9", "10"}
            and self.fasta.index[c][0] > 2 * TISP_SEQUENCE_LENGTH
        ]
        weights = np.asarray(
            [self.fasta.index[c][0] - 2 * TISP_SEQUENCE_LENGTH for c in self.chroms],
            dtype=np.float64,
        )
        self.weights = weights / weights.sum()

    def __iter__(self):
        # The public code seeds NumPy immediately before constructing the
        # sampler and uses an effectively endless random stream.
        rng = np.random.RandomState(TRAIN_SEED)
        while True:
            chrom = str(rng.choice(self.chroms, p=self.weights))
            chrom_len = int(self.fasta.index[chrom][0])
            center = int(rng.randint(TISP_SEQUENCE_LENGTH, chrom_len - TISP_SEQUENCE_LENGTH))
            start = center - TISP_SEQUENCE_LENGTH // 2
            end = start + TISP_SEQUENCE_LENGTH
            if self.blacklist.overlaps(chrom, start, end):
                continue
            sequence = self.fasta.fetch(chrom, start, end)
            if len(sequence) != TISP_SEQUENCE_LENGTH:
                continue
            if sequence.count("N") / TISP_SEQUENCE_LENGTH > 0.30:
                continue
            yield {"input_ids": _ids(sequence), "target": torch.from_numpy(self.tracks.get(chrom, start, end))}


class OriginalSlidingTISP(Dataset):
    """The public 100-kb/50-kb sliding windows, without end-aligned extras."""

    def __init__(self, root: str, chroms: list[str]):
        self.root = Path(root)
        self.fasta = IndexedFasta(
            self.root / "tisp/seqs/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
        )
        self.tracks = TISPTracks(root)
        self.windows: list[tuple[str, int]] = []
        for requested in chroms:
            chrom = _resolve_chrom(self.fasta.index, requested)
            length = int(self.fasta.index[chrom][0])
            self.windows.extend(
                (chrom, start)
                for start in range(0, length - TISP_SEQUENCE_LENGTH + 1, TISP_WINDOW_STEP)
            )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        chrom, start = self.windows[int(index)]
        sequence = self.fasta.fetch(chrom, start, start + TISP_SEQUENCE_LENGTH)
        return {
            "input_ids": _ids(sequence),
            "target": torch.from_numpy(
                self.tracks.get(chrom, start, start + TISP_SEQUENCE_LENGTH)
            ),
        }


class _PCC:
    def __init__(self):
        self.n = 0.0
        self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0

    def update(self, x: np.ndarray, y: np.ndarray, valid: np.ndarray | None = None):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        if valid is not None:
            mask &= valid
        x, y = x[mask], y[mask]
        self.n += x.size
        self.sx += x.sum(); self.sy += y.sum()
        self.sxx += np.square(x).sum(); self.syy += np.square(y).sum(); self.sxy += (x * y).sum()

    def value(self) -> float:
        if self.n <= 1:
            return float("nan")
        vx = self.sxx - self.sx * self.sx / self.n
        vy = self.syy - self.sy * self.sy / self.n
        cov = self.sxy - self.sx * self.sy / self.n
        return float(cov / math.sqrt(vx * vy)) if vx > 0 and vy > 0 else float("nan")


def _transform(values: np.ndarray, name: str) -> np.ndarray:
    if name == "cage_like":
        return np.log10(np.maximum(np.power(10.0, values) - 1.0 + 0.1, 1e-12))
    return values / math.log(10.0)


def _update_pcc(stats: dict[str, _PCC], pred: np.ndarray, target: np.ndarray, valid: np.ndarray):
    for name, (plus, minus, transform) in TISP_ASSAY_GROUPS.items():
        stats[name].update(_transform(pred[:, plus], transform), _transform(target[:, plus], transform), valid)
        stats[name].update(_transform(pred[:, minus], transform), _transform(target[:, minus], transform), valid)


def _metrics(stats: dict[str, _PCC]) -> dict[str, float]:
    values = {f"{name}_pcc": stat.value() for name, stat in stats.items()}
    finite = [value for value in values.values() if np.isfinite(value)]
    values["avg_pcc"] = float(np.mean(finite)) if finite else float("nan")
    return values


def _official_valid_mask(sequence: str) -> np.ndarray:
    """Mask each N base and its 500-bp neighbourhood as in process_tisp.py."""
    invalid = np.frombuffer(sequence.upper().encode("ascii"), dtype=np.uint8) == ord("N")
    positions = np.flatnonzero(invalid)
    if positions.size == 0:
        return np.ones(invalid.size, dtype=bool)
    # Difference-array form of the 1001-bp convolution.  The direct
    # np.convolve implementation is quadratic in the 50-kb segment length
    # here and made the final chromosome evaluation needlessly slow.
    diff = np.zeros(invalid.size + 1, dtype=np.int32)
    starts = np.maximum(positions - 500, 0)
    ends = np.minimum(positions + 501, invalid.size)
    np.add.at(diff, starts, 1)
    np.add.at(diff, ends, -1)
    return np.cumsum(diff[:-1]) == 0


@torch.inference_mode()
def evaluate_validation(model, loader, device, max_batches: int = VALIDATION_BATCHES):
    model.eval()
    total_loss = 0.0
    batches = 0; metric_sum = 0.0; metric_count = 0
    center = slice(TRIM, TISP_SEQUENCE_LENGTH - TRIM)
    index = 0
    dataset = loader.dataset
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        target = batch["target"].to(device)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=device.type == "cuda" and bool(getattr(model, "use_bf16", False)),
        ):
            output = model(input_ids, target)
        pred = output["pred"].float().cpu().numpy()[:, center, :]
        y = target.float().cpu().numpy()[:, center, :]
        valid = np.ones(pred.shape[0:2], dtype=bool)
        for row in range(pred.shape[0]):
            chrom, start = dataset.windows[index]
            index += 1
            valid[row] = _official_valid_mask(
                dataset.fasta.fetch(chrom, start + TRIM, start + TISP_SEQUENCE_LENGTH - TRIM)
            )
        # This is the public DNALongBench ``pearson_corr_coef`` metric:
        # Pearson is computed along sequence positions independently for all
        # ten tracks, then averaged over tracks and examples.
        for row in range(pred.shape[0]):
            per_track = []
            for channel in range(pred.shape[-1]):
                x = pred[row, :, channel][valid[row]]
                yy = y[row, :, channel][valid[row]]
                if x.size < 2:
                    continue
                x = x - x.mean(); yy = yy - yy.mean()
                denom = np.linalg.norm(x) * np.linalg.norm(yy)
                if denom > 0:
                    per_track.append(float(np.dot(x, yy) / denom))
            if per_track:
                metric_sum += float(np.mean(per_track)); metric_count += 1
        total_loss += float(output["loss"].float().item()); batches += 1
        if batches >= max_batches:
            break
    result = {"avg_pcc": metric_sum / max(1, metric_count),
              "metric": "DNALongBench pearson_corr_coef over 10 channels"}
    result["loss"] = total_loss / max(1, batches)
    result["batches"] = batches
    return result


@torch.inference_mode()
def evaluate_test_chromosomes(model, loader, device, dataset: OriginalSlidingTISP):
    """Evaluate center segments with the public neighbouring-window average."""
    model.eval()
    stats = {name: _PCC() for name in TISP_ASSAY_GROUPS}
    previous: dict[str, tuple[int, np.ndarray, np.ndarray, str]] = {}
    total_loss = 0.0; batches = 0; index = 0
    center = slice(TRIM, TISP_SEQUENCE_LENGTH - TRIM)
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=device.type == "cuda" and bool(getattr(model, "use_bf16", False)),
            ):
                output = model(batch["input_ids"].to(device), batch["target"].to(device))
            pred = output["pred"].float().cpu().numpy()
            target = batch["target"].numpy()
            total_loss += float(output["loss"].float().item()); batches += 1
            for row in range(pred.shape[0]):
                chrom, start = dataset.windows[index]
                index += 1
                current = (int(start), pred[row, center, :], target[row, center, :], "")
                old = previous.get(chrom)
                if old is not None and int(start) == old[0] + TISP_WINDOW_STEP:
                    # This is the published process_tisp.py rule: each
                    # central 50-kb segment is the average of two adjacent
                    # 100-kb predictions, counted once.
                    segment_pred = (old[1] + current[1]) * 0.5
                    segment_target = old[2]
                    valid = _official_valid_mask(
                        dataset.fasta.fetch(
                            chrom, old[0] + TRIM, old[0] + TISP_SEQUENCE_LENGTH - TRIM
                        )
                    )
                    _update_pcc(stats, segment_pred, segment_target, valid)
                previous[chrom] = current
    result = _metrics(stats)
    result["loss"] = total_loss / max(1, batches)
    result["windows"] = index
    return result


def _cosine_lr(step: int, total_steps: int, base_lr: float) -> float:
    warmup = max(1, int(total_steps * 0.01))
    if step < warmup:
        return WARMUP_LR_INIT + (base_lr - WARMUP_LR_INIT) * (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total_steps - warmup)))
    return 0.1 * base_lr + 0.5 * (base_lr - 0.1 * base_lr) * (1.0 + math.cos(math.pi * progress))


def _seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    # NTv3's FP32 memory footprint permits four 100-kb examples on an H100;
    # accumulation keeps the published global batch exactly 16.
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--all-bf16", action="store_true",
                   help="use BF16 parameters, compute, adapter, and loss")
    p.add_argument("--ntv3-mixed-precision", action="store_true",
                   help="use BF16 NTv3 body compute with FP32 sensitive modules, adapter, and loss")
    p.add_argument("--vericomp-layernorm-head", action="store_true",
                   help="use the historical VeriComp LayerNorm + Linear head, independently of precision")
    p.add_argument("--grad-accumulation", type=int, default=4)
    p.add_argument("--train-steps", type=int, default=25000)
    p.add_argument("--eval-every", type=int, default=VALIDATION_INTERVAL)
    p.add_argument("--lr", type=float, default=6e-4,
                   help="Caduceus DNALongBench TISP LR; NTv3 has no published TISP LR")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="cosine",
                   help="learning-rate schedule; constant disables warmup and decay")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args(); _seed(args.seed)
    if args.all_bf16 and args.ntv3_mixed_precision:
        raise ValueError("--all-bf16 and --ntv3-mixed-precision are mutually exclusive")
    device = torch.device(args.device)
    root = Path(args.data_root).resolve()
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)

    # Use the same random-window training distribution and chromosome-level
    # evaluator as the successful VeriComp-TISP configuration.
    train = TISPRandomDataset(str(root), 0)
    valid = TISPWindowDataset(str(root), ["10"])
    test = TISPWindowDataset(str(root), ["8", "9"])
    train_loader = DataLoader(train, batch_size=args.batch_size, num_workers=0)
    valid_loader = DataLoader(valid, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
    # Keep the historical GeneZip-style LayerNorm + Linear head when running
    # the all-BF16 precision comparison.  The strict DNALongBench head is a
    # separate protocol choice, not a precision choice.
    strict_dnalongbench = not (
        args.ntv3_mixed_precision or args.all_bf16 or args.vericomp_layernorm_head
    )
    full_fp32 = not (args.ntv3_mixed_precision or args.all_bf16)
    model = NTv3TISPModel(
        args.model_path, device, all_bf16=args.all_bf16, full_fp32=full_fp32,
        strict_dnalongbench=strict_dnalongbench,
    )
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999), eps=1e-8,
    )
    config = {
        "start_time_utc": datetime.now(timezone.utc).isoformat(),
        "task": "TISP", "model": "NTv3-100M-pre",
        "protocol": "VeriComp_TISP_GeneZip_chromosome_eval",
        "source_repository": "https://github.com/wenduocheng/DNALongBench",
        "paper": "https://www.nature.com/articles/s41467-025-65077-4",
        "data_root": str(root), "sequence_length": TISP_SEQUENCE_LENGTH,
        "center_scored_length": TISP_WINDOW_STEP,
        "train_chromosomes": "all except 8, 9, 10", "validation": "chromosome 10",
        "test": ["chromosome 8", "chromosome 9"],
        "train_sampling": "TISPRandomDataset random intervals, seed 3, N fraction <= 0.30",
        "batch_size": args.batch_size, "gradient_accumulation": args.grad_accumulation,
        "effective_batch_size": args.batch_size * args.grad_accumulation,
        "train_steps": args.train_steps, "validation_every_steps": args.eval_every,
        "validation_limit_batches": None,
        "learning_rate": args.lr,
        "learning_rate_note": "User-specified learning rate; NTv3 has no published TISP learning rate",
        "optimizer": "AdamW", "weight_decay": args.weight_decay,
        "betas": [0.9, 0.999], "eps": 1e-8, "gradient_clip": args.max_grad_norm,
        "scheduler": "constant" if args.lr_schedule == "constant" else "cosine warmup timm-equivalent, 1% warmup, lr_min=0.1*lr",
        "precision": (
            "BF16 all modules and loss" if args.all_bf16 else
            "BF16 backbone compute; FP32 sensitive modules, adapter, and loss"
            if args.ntv3_mixed_precision else "FP32 all modules and loss"
        ),
        "decoder": (
            "single Linear(hidden, 10), no LayerNorm"
            if strict_dnalongbench else
            "LayerNorm + Linear(hidden, 10)"
        ),
        "ntv3_strict_dnalongbench": strict_dnalongbench,
        "feature": "outputs.hidden_states[-1]",
        "loss": "pseudo_poisson_kl",
        "evaluation": "GeneZip-compatible chromosome-level 50-kb centre de-duplication and five strand-paired PCCs",
        "validation_best": "validation average PCC",
        "dataset_sizes": {"valid_windows": len(valid), "test_windows": len(test)},
        "arguments": vars(args),
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2), flush=True)

    best = -math.inf; history = []; optimizer.zero_grad(set_to_none=True)
    iterator = iter(train_loader); started = time.perf_counter()
    for step in range(1, args.train_steps + 1):
        for micro in range(args.grad_accumulation):
            batch = next(iterator)
            output = model(batch["input_ids"].to(device), batch["target"].to(device))
            (output["loss"] / args.grad_accumulation).backward()
        torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        lr = args.lr if args.lr_schedule == "constant" else _cosine_lr(step - 1, args.train_steps, args.lr)
        for group in optimizer.param_groups: group["lr"] = lr
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % args.log_every == 0:
            print(json.dumps({"step": step, "train_loss": float(output["loss"].detach()),
                              "lr": lr, "step_s": (time.perf_counter() - started) / step}), flush=True)
        if step % args.eval_every == 0 or step == args.train_steps:
            valid_metrics = evaluate_tisp_chromosome(model, valid_loader, device)
            record = {"step": step, "valid": valid_metrics}; history.append(record)
            print(json.dumps(record), flush=True)
            if np.isfinite(valid_metrics["avg_pcc"]) and valid_metrics["avg_pcc"] > best:
                best = valid_metrics["avg_pcc"]
                torch.save({"model": model.state_dict(), "step": step,
                            "valid": valid_metrics, "config": config}, out / "best.pt")
        if step % 300 == 0:
            torch.save({"model": model.state_dict(), "step": step, "config": config}, out / f"step_{step:05d}.pt")

    # Report both the final checkpoint and the checkpoint selected only from
    # full chr10 validation.  Test chromosomes are never used during training.
    final_test = evaluate_tisp_chromosome(model, test_loader, device)
    # Preserve the final model before loading validation-best for its test run.
    torch.save({"model": model.state_dict(), "step": args.train_steps,
                "test_final": final_test, "config": config}, out / "last.pt")
    best_test = None
    best_checkpoint = out / "best.pt"
    if best_checkpoint.exists():
        payload = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(payload["model"])
        best_test = evaluate_tisp_chromosome(model, test_loader, device)
    result = {"task": "TISP", "model": "NTv3-100M-pre", "valid_best_avg_pcc": best,
              "test_final": final_test, "test_validation_best": best_test,
              "history": history}
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output_dir": str(out), **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
