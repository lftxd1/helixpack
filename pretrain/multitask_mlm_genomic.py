"""Joint MLM and genomic annotation/signal post-training for VeriComp.

Each optimizer update uses one MLM batch and one genomic window batch. The
genomic batch supplies three independent targets (GENCODE structure, H3K27ac,
and ATAC signal) over the central 100 kb of a 128-Ki-base input. Detached
per-batch coefficients make each weighted supervised loss equal 0.3 times
the MLM cross-entropy; MoE router regularization remains a separate term.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from pretrain.multitask_mlm_tisp import (
    _atomic_json,
    _atomic_save,
    _load_config_and_state,
    _reduce_values,
    _seed,
)
from pretrain.run_config import print_resolved_config, write_run_config
from pretrain.train import (
    AsyncLogWriter,
    build_optimizer,
    make_loader,
    masked_accuracy,
    setup_distributed,
)
from vericomp import VeriCompConfig
from vericomp.backbone import reverse_complement_ids
from vericomp.data import BEDGenomeDataset, IndexedFasta


DEFAULT_CHECKPOINT = (
    "/gpfs_data/Projects/HelixPack/weights/pretrain/20260814/"
    "vericomp-residual-skip-30k/checkpoint-010000.pt"
)
DEFAULT_MLM_FASTA = "/gpfs_data/Projects/data/hg38/hg38.ml.fa"
DEFAULT_MLM_BED = "/gpfs_data/Projects/data/hg38/human-sequences.bed"
DEFAULT_GFF3 = (
    "/gpfs_data/Projects/HelixPack/data/auxiliary/gencode_v50/"
    "gencode.v50.annotation.sorted.gff3.gz"
)
DEFAULT_H3K27AC = (
    "/gpfs_data/Projects/HelixPack/data/auxiliary/encode_h3k27ac/"
    "ENCFF091FMS.bigWig"
)
DEFAULT_ATAC = (
    "/gpfs_data/Projects/HelixPack/data/auxiliary/encode_atac_heart/"
    "ENCFF073QRI.bigWig"
)
DEFAULT_CONSERVATION = (
    "/gpfs_data/Projects/HelixPack/data/auxiliary/conservation/phastCons100way/"
    "hg38.phastCons100way.bw"
)
CONTEXT_LENGTH = 131_072
TARGET_LENGTH = 100_000
TARGET_OFFSET = (CONTEXT_LENGTH - TARGET_LENGTH) // 2
ANNOTATION_NAMES = ("gene", "exon", "CDS", "five_prime_UTR", "three_prime_UTR")
GFF_FEATURES = {
    "gene": "gene",
    "exon": "exon",
    "CDS": "CDS",
    "five_prime_UTR": "five_prime_UTR",
    "three_prime_UTR": "three_prime_UTR",
}
TRAIN_HELDOUT = {"chr8", "chr9", "chr10"}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--mlm-fasta", default=DEFAULT_MLM_FASTA)
    p.add_argument("--mlm-bed", default=DEFAULT_MLM_BED)
    p.add_argument("--gff3", default=DEFAULT_GFF3)
    p.add_argument("--h3k27ac-bigwig", default=DEFAULT_H3K27AC)
    p.add_argument("--atac-bigwig", default=DEFAULT_ATAC)
    p.add_argument("--conservation-bigwig", default=DEFAULT_CONSERVATION)
    p.add_argument(
        "--output-dir",
        default=(
            "weights/pretrain/20260921/"
            "vericomp-mlm-gencode-h3k27ac-atac-128k-ratio0p3-b4-30k"
        ),
    )
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--eval-every", type=int, default=1_000)
    p.add_argument("--validation-samples", type=int, default=32)
    p.add_argument("--test-samples", type=int, default=64)
    p.add_argument("--mlm-batch-size", type=int, default=4)
    p.add_argument("--genomic-batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--mlm-workers", type=int, default=2)
    p.add_argument(
        "--profile-data-steps", type=int, default=100,
        help="synchronize and separately time data loading and model compute for the first N updates",
    )
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-end", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--router-loss-coef", type=float, default=None,
                   help="defaults to the router auxiliary coefficient in the source checkpoint")
    p.add_argument("--target-loss-ratio", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--smoke-test", action="store_true",
                   help="run one optimizer update and one inference pass, without saving checkpoints")
    return p.parse_args()


def _primary_chromosome(chrom):
    return chrom in {*(f"chr{i}" for i in range(1, 23)), "chrX", "chrY"}


def ensure_annotation_cache(gff3_path: Path, cache_path: Path) -> None:
    """Create a compact sorted interval cache from the exact GENCODE GFF3."""
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return
    values = {
        (chrom, label): ([], [])
        for chrom in [*(f"chr{i}" for i in range(1, 23)), "chrX", "chrY"]
        for label in ANNOTATION_NAMES
    }
    with gzip.open(gff3_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line or line[0] == "#":
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5 or not _primary_chromosome(fields[0]):
                continue
            label = fields[2]
            if label not in GFF_FEATURES:
                continue
            try:
                start, end = int(fields[3]) - 1, int(fields[4])
            except ValueError:
                continue
            if end <= start:
                continue
            starts, ends = values[(fields[0], label)]
            starts.append(start)
            ends.append(end)
    arrays = {}
    for (chrom, label), (starts, ends) in values.items():
        if starts:
            order = np.argsort(starts, kind="stable")
            sorted_starts = np.asarray(starts, dtype=np.int32)[order]
            sorted_ends = np.asarray(ends, dtype=np.int32)[order]
            arrays[f"{chrom}__{label}__start"] = sorted_starts
            arrays[f"{chrom}__{label}__end"] = sorted_ends
            arrays[f"{chrom}__{label}__prefix_end"] = np.maximum.accumulate(sorted_ends)
        else:
            arrays[f"{chrom}__{label}__start"] = np.empty(0, dtype=np.int32)
            arrays[f"{chrom}__{label}__end"] = np.empty(0, dtype=np.int32)
            arrays[f"{chrom}__{label}__prefix_end"] = np.empty(0, dtype=np.int32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=cache_path.name + ".", suffix=".npz",
                               dir=cache_path.parent)
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, cache_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class GenomicAuxDataset(Dataset):
    """GRCh38 windows with GENCODE, ENCODE signals, and phastCons labels."""

    def __init__(self, fasta_path, gff3_path, gff3_cache, h3_path, atac_path,
                 conservation_path,
                 *, split: str, context_length: int, seed: int,
                 sample_count: int = 0):
        import pyBigWig

        self.fasta = IndexedFasta(fasta_path)
        self.gff3_path = Path(gff3_path)
        self.gff3_cache = Path(gff3_cache)
        with np.load(self.gff3_cache, allow_pickle=False) as archive:
            self.annotations = {
                chrom: {
                    label: (
                        archive[f"{chrom}__{label}__start"],
                        archive[f"{chrom}__{label}__end"],
                        archive[f"{chrom}__{label}__prefix_end"],
                    )
                    for label in ANNOTATION_NAMES
                }
                for chrom in [*(f"chr{i}" for i in range(1, 23)), "chrX", "chrY"]
            }
        self.h3_path, self.atac_path = str(h3_path), str(atac_path)
        self.conservation_path = str(conservation_path)
        self.h3 = pyBigWig.open(self.h3_path)
        self.atac = pyBigWig.open(self.atac_path)
        self.conservation = pyBigWig.open(self.conservation_path)
        if self.h3 is None or self.atac is None or self.conservation is None:
            raise RuntimeError("could not open one of the signal/conservation bigWigs")
        if self.h3.chroms() != self.atac.chroms():
            # ENCODE signal tracks may include different non-primary contigs.
            # Only require the primary assembly sequence lengths to agree.
            for chrom in [*(f"chr{i}" for i in range(1, 23)), "chrX"]:
                if (self.h3.chroms().get(chrom) != self.atac.chroms().get(chrom)
                        or self.h3.chroms().get(chrom) != self.fasta.index.get(chrom, (None,))[0]):
                    raise ValueError(f"GRCh38 chromosome length mismatch at {chrom}")
        for chrom in [*(f"chr{i}" for i in range(1, 23)), "chrX"]:
            if (self.conservation.chroms().get(chrom)
                    != self.fasta.index.get(chrom, (None,))[0]):
                raise ValueError(f"conservation GRCh38 chromosome length mismatch at {chrom}")
        self.context_length = int(context_length)
        self.seed = int(seed)
        self.split = split
        if split == "train":
            eligible = [c for c in self.fasta.index if _primary_chromosome(c)
                        and c not in TRAIN_HELDOUT]
        elif split == "valid":
            eligible = ["chr10"]
        elif split == "test":
            eligible = ["chr8", "chr9"]
        else:
            raise ValueError(f"unknown split: {split}")
        self.chroms = [
            c for c in eligible
            if c in self.fasta.index
            and c in self.h3.chroms()
            and c in self.atac.chroms()
            and c in self.conservation.chroms()
            and self.fasta.index[c][0] > self.context_length
        ]
        if not self.chroms:
            raise ValueError(f"no chromosomes available for split={split}")
        chrom_lengths = np.asarray(
            [self.fasta.index[c][0] - self.context_length for c in self.chroms],
            dtype=np.float64,
        )
        self.chrom_probabilities = chrom_lengths / chrom_lengths.sum()
        self.fixed_windows = []
        if split != "train":
            remaining = max(0, int(sample_count))
            per_chrom = max(1, math.ceil(remaining / len(self.chroms)))
            for chrom in self.chroms:
                chrom_len = self.fasta.index[chrom][0]
                max_start = chrom_len - self.context_length
                count = min(
                    per_chrom,
                    max(1, math.ceil(chrom_len / self.context_length)),
                )
                starts = np.linspace(0, max_start, num=count, dtype=np.int64)
                self.fixed_windows.extend((chrom, int(s)) for s in starts)
            self.fixed_windows = self.fixed_windows[:remaining]
        else:
            self.fixed_windows = None

    def __len__(self):
        return len(self.fixed_windows) if self.fixed_windows is not None else 1_000_000

    @staticmethod
    def _ids(sequence):
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        return torch.tensor([mapping.get(base, 5) for base in sequence], dtype=torch.long)

    def _annotation_target(self, chrom, start, end):
        target = np.zeros((end - start, len(ANNOTATION_NAMES)), dtype=np.float32)
        for channel, label in enumerate(ANNOTATION_NAMES):
            starts, ends, prefix_end = self.annotations[chrom][label]
            stop = int(np.searchsorted(starts, end, side="left"))
            if stop <= 0:
                continue
            begin = int(np.searchsorted(prefix_end[:stop], start, side="right"))
            if begin >= stop:
                continue
            candidates = np.flatnonzero(ends[begin:stop] > start) + begin
            for index in candidates:
                left = max(start, int(starts[index])) - start
                right = min(end, int(ends[index])) - start
                if right > left:
                    target[left:right, channel] = 1.0
        return target

    @staticmethod
    def _signal_target(bigwig, chrom, start, end):
        values = bigwig.values(chrom, start, end, numpy=True)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = np.clip(values, 0.0, 1000.0)
        return np.log1p(values).astype(np.float32, copy=False)

    @staticmethod
    def _conservation_target(bigwig, chrom, start, end):
        values = np.asarray(bigwig.values(chrom, start, end, numpy=True), dtype=np.float32)
        valid = np.isfinite(values)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = np.clip(values, 0.0, 1.0)
        return values.astype(np.float32, copy=False), valid

    def __getitem__(self, index):
        if self.fixed_windows is None:
            rng = np.random.default_rng(self.seed + int(index) * 104729)
            chrom = str(rng.choice(self.chroms, p=self.chrom_probabilities))
            chrom_len = self.fasta.index[chrom][0]
            start = int(rng.integers(0, chrom_len - self.context_length + 1))
        else:
            chrom, start = self.fixed_windows[int(index)]
        seq = self.fasta.fetch(chrom, start, start + self.context_length)
        if len(seq) != self.context_length:
            raise RuntimeError(f"short FASTA window at {chrom}:{start}")
        ids = self._ids(seq)
        target_start = start + TARGET_OFFSET
        target_end = target_start + TARGET_LENGTH
        conservation, conservation_valid = self._conservation_target(
            self.conservation, chrom, target_start, target_end
        )
        return {
            "input_ids": ids,
            "annotation": torch.from_numpy(
                self._annotation_target(chrom, target_start, target_end)
            ),
            "h3k27ac": torch.from_numpy(
                self._signal_target(self.h3, chrom, target_start, target_end)
            ),
            "atac": torch.from_numpy(
                self._signal_target(self.atac, chrom, target_start, target_end)
            ),
            "conservation": torch.from_numpy(conservation),
            "conservation_valid": torch.from_numpy(conservation_valid),
        }


class MultiTaskGenomicModel(nn.Module):
    TASKS = ("gencode", "h3k27ac", "atac", "conservation")

    def __init__(self, config: VeriCompConfig, target_ratio: float,
                 router_loss_coef: float):
        super().__init__()
        from vericomp.modeling import VeriCompPretrainModel

        self.backbone = VeriCompPretrainModel(config)
        self.target_ratio = float(target_ratio)
        self.router_loss_coef = float(router_loss_coef)
        width = config.d_model
        self.annotation_norm = nn.LayerNorm(width)
        self.annotation_head = nn.Linear(width, len(ANNOTATION_NAMES))
        self.h3_norm = nn.LayerNorm(width)
        self.h3_head = nn.Linear(width, 1)
        self.atac_norm = nn.LayerNorm(width)
        self.atac_head = nn.Linear(width, 1)
        self.conservation_norm = nn.LayerNorm(width)
        self.conservation_head = nn.Linear(width, 1)

    @staticmethod
    def _global_mean(loss):
        value = loss.detach().to(torch.float64)
        if dist.is_initialized() and torch.is_grad_enabled():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value /= dist.get_world_size()
        return value.to(dtype=loss.dtype)

    @staticmethod
    def _balanced_annotation_loss(logits, target, valid):
        logits = logits.float()
        target = target.float()
        valid = valid.unsqueeze(-1).expand_as(target)
        per_channel = []
        for channel in range(target.shape[-1]):
            y = target[..., channel]
            positive = valid[..., channel] & y.gt(0.5)
            negative = valid[..., channel] & ~y.gt(0.5)
            pos_loss = F.softplus(-logits[..., channel])[positive].mean() if positive.any() else None
            neg_loss = F.softplus(logits[..., channel])[negative].mean() if negative.any() else None
            if pos_loss is not None and neg_loss is not None:
                per_channel.append(0.5 * (pos_loss + neg_loss))
            elif pos_loss is not None:
                per_channel.append(pos_loss)
            elif neg_loss is not None:
                per_channel.append(neg_loss)
        if not per_channel:
            return logits.sum() * 0.0
        return torch.stack(per_channel).mean()

    @staticmethod
    def _masked_huber(pred, target, valid):
        element_loss = F.smooth_l1_loss(
            pred.float(), target.float(), reduction="none", beta=0.5
        )
        mask = valid.to(dtype=element_loss.dtype)
        return (element_loss * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(self, mlm_batch, genomic_batch):
        mlm_output = self.backbone(
            input_ids_pos=mlm_batch["input_ids_pos"],
            input_ids_rc=mlm_batch["input_ids_rc"],
            labels=mlm_batch["labels"],
        )
        sequence = genomic_batch["input_ids"]
        genomic_output = self.backbone(
            input_ids_pos=sequence,
            input_ids_rc=reverse_complement_ids(sequence),
            return_decoder_feature=True,
        )
        features = genomic_output["decoder_feature"]
        if features.shape[1] != sequence.shape[1]:
            features = F.interpolate(
                features.transpose(1, 2), size=sequence.shape[1],
                mode="linear", align_corners=False,
            ).transpose(1, 2)
        center = features[:, TARGET_OFFSET:TARGET_OFFSET + TARGET_LENGTH]
        valid = sequence[:, TARGET_OFFSET:TARGET_OFFSET + TARGET_LENGTH].ne(5)
        logits_annotation = self.annotation_head(self.annotation_norm(center))
        pred_h3 = self.h3_head(self.h3_norm(center)).squeeze(-1)
        pred_atac = self.atac_head(self.atac_norm(center)).squeeze(-1)
        pred_conservation = self.conservation_head(
            self.conservation_norm(center)
        ).squeeze(-1)
        raw = {
            "gencode": self._balanced_annotation_loss(
                logits_annotation, genomic_batch["annotation"], valid
            ),
            "h3k27ac": self._masked_huber(
                pred_h3, genomic_batch["h3k27ac"], valid
            ),
            "atac": self._masked_huber(
                pred_atac, genomic_batch["atac"], valid
            ),
            "conservation": self._masked_huber(
                pred_conservation, genomic_batch["conservation"],
                valid & genomic_batch["conservation_valid"].bool(),
            ),
        }
        mlm_loss = mlm_output["loss_mlm"].float()
        weights = {}
        weighted = {}
        ratios = {}
        for name, task_loss in raw.items():
            global_mlm = self._global_mean(mlm_loss)
            global_task = self._global_mean(task_loss)
            if (not bool(torch.isfinite(global_mlm))
                    or not bool(torch.isfinite(global_task))
                    or float(global_mlm) <= 0 or float(global_task) <= 0):
                raise RuntimeError(
                    f"cannot dynamically balance {name}: "
                    f"MLM={global_mlm.item()} task={global_task.item()}"
                )
            weights[name] = (
                self.target_ratio * global_mlm / global_task
            ).detach()
            weighted[name] = weights[name] * task_loss
            ratios[name] = (
                weights[name] * global_task / global_mlm.clamp_min(1e-12)
            ).detach()
        # Router regularization is not assigned to any one supervised task.
        router_aux = 0.5 * (
            mlm_output["loss_moe"].float()
            + genomic_output["loss_router"].float()
            * self.router_loss_coef
        )
        total = mlm_loss + sum(weighted.values()) + router_aux
        return {
            "loss": total,
            "mlm_loss": mlm_loss,
            "router_aux": router_aux,
            "task_losses": raw,
            "task_weights": weights,
            "weighted_tasks": weighted,
            "task_ratios": ratios,
            "mlm": mlm_output,
        }


def _read_model_config(checkpoint, activation_checkpointing):
    source = checkpoint.get("model_config")
    if isinstance(source, dict):
        from dataclasses import fields
        valid = {f.name for f in fields(VeriCompConfig)}
        values = {k: v for k, v in source.items() if k in valid}
    else:
        values = {}
        source_args = checkpoint.get("args", {})
        for name in ("num_layers", "allocation_mode", "strict_uncompressed_short",
                     "duplex_mode"):
            if name in source_args:
                values[name] = source_args[name]
    values.update(force_flash_attention=True,
                  activation_checkpointing=bool(activation_checkpointing))
    return VeriCompConfig(**values)


def _load_pretrained_backbone(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    target = model.state_dict()
    compatible = {}
    for name, value in state.items():
        candidates = (name, name[9:]) if name.startswith("backbone.") else (
            name, f"backbone.{name}"
        )
        for candidate in candidates:
            if candidate in target and tuple(target[candidate].shape) == tuple(value.shape):
                compatible[candidate] = value
                break
    backbone_keys = {f"backbone.{name}" for name in model.backbone.state_dict()}
    missing_backbone = sorted(backbone_keys - set(compatible))
    if missing_backbone:
        unexpected = sorted(set(state) - set(compatible))
        raise RuntimeError(
            "checkpoint/backbone mismatch: "
            f"loaded {len(compatible)}/{len(backbone_keys)} backbone tensors, "
            f"missing={missing_backbone[:8]}, unexpected={unexpected[:8]}"
        )
    model.load_state_dict(compatible, strict=False)
    return len(compatible)


def _save_checkpoint(model, output, args, step, valid, best_score, *, best):
    payload = {
        "task": "MLM+GENCODE+H3K27ac+ATAC+phastCons",
        "model": model.state_dict(),
        "model_config": model.backbone.config.__dict__,
        "step": int(step),
        "validation": valid,
        "best_validation_objective": float(best_score),
        "target_supervised_to_mlm_loss_ratio": float(args.target_loss_ratio),
        "arguments": vars(args),
    }
    if best:
        _atomic_save(payload, output / "best.pt")
    _atomic_save(payload, output / "last.pt")


def _evaluate(model, mlm_loader, genomic_loader, device):
    model.eval()
    values = []
    task_values = {name: [] for name in model.TASKS}
    mlm_values = []
    with torch.inference_mode():
        for mlm_batch, genomic_batch in zip(mlm_loader, genomic_loader):
            mlm_batch = {k: v.to(device, non_blocking=True) for k, v in mlm_batch.items()}
            genomic_batch = {k: v.to(device, non_blocking=True) for k, v in genomic_batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(mlm_batch, genomic_batch)
            values.append(float(output["loss"].float()))
            mlm_values.append(float(output["mlm_loss"].float()))
            for name in model.TASKS:
                task_values[name].append(float(output["task_losses"][name].float()))
    result = {
        "composite_objective": float(np.mean(values)),
        "mlm_loss": float(np.mean(mlm_values)),
        "task_losses": {name: float(np.mean(v)) for name, v in task_values.items()},
        "chromosome_split": "chr10 validation",
        "windows": len(values),
    }
    model.train()
    return result


def _move(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def main():
    args = parse_args()
    distributed, rank, world, device = setup_distributed()
    writer = AsyncLogWriter(rank == 0)
    try:
        if not torch.cuda.is_available() or device.type != "cuda":
            raise RuntimeError("genomic multi-task post-training requires CUDA")
        if world != 4:
            raise RuntimeError(f"this launch expects four GPUs, got {world}")
        if (args.target_loss_ratio <= 0 or args.steps < 1 or args.eval_every < 1
                or args.profile_data_steps < 0):
            raise ValueError("loss ratio, steps, and evaluation interval must be positive")
        for path in (args.checkpoint, args.mlm_fasta, args.mlm_bed, args.gff3,
                     args.h3k27ac_bigwig, args.atac_bigwig,
                     args.conservation_bigwig):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        _seed(args.seed, rank)
        output = Path(args.output_dir).resolve()
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
        cache = Path(args.gff3).with_name("gencode_v50_primary_intervals_v1.npz")
        if rank == 0:
            ensure_annotation_cache(Path(args.gff3), cache)
        if distributed:
            dist.barrier()

        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = _read_model_config(checkpoint, args.activation_checkpointing)
        del checkpoint
        model = MultiTaskGenomicModel(
            config, target_ratio=args.target_loss_ratio,
            router_loss_coef=(
                config.router_aux_loss_coef
                if args.router_loss_coef is None else args.router_loss_coef
            ),
        ).to(device)
        loaded = _load_pretrained_backbone(model, args.checkpoint)
        if hasattr(model.backbone, "use_bf16"):
            model.backbone.use_bf16 = True
        ddp = DDP(
            model, device_ids=[device.index], output_device=device.index,
            gradient_as_bucket_view=True, find_unused_parameters=False,
        )
        optimizer = build_optimizer(
            ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True
        )

        mlm_train = BEDGenomeDataset(
            args.mlm_bed, args.mlm_fasta, CONTEXT_LENGTH,
            sequence_lengths=[CONTEXT_LENGTH], split="train", seed=args.seed,
            random_window_sampling=True,
        )
        mlm_loader, mlm_sampler = make_loader(
            mlm_train, world_size=world, rank=rank,
            batch_sizes={CONTEXT_LENGTH: args.mlm_batch_size},
            seed=args.seed, shuffle=True, drop_last=True, workers=args.mlm_workers,
            equal_length_sampling=True,
        )
        mlm_sampler.set_epoch(0)
        mlm_iter = iter(mlm_loader)
        train_ds = GenomicAuxDataset(
            args.mlm_fasta, args.gff3, cache, args.h3k27ac_bigwig,
            args.atac_bigwig, args.conservation_bigwig,
            split="train", context_length=CONTEXT_LENGTH,
            seed=args.seed + rank * 10_007,
        )
        genomic_loader = DataLoader(
            train_ds, batch_size=args.genomic_batch_size, shuffle=False,
            num_workers=0, pin_memory=True, drop_last=True,
        )
        genomic_iter = iter(genomic_loader)
        params = sum(p.numel() for p in model.parameters())

        valid_genomic = valid_mlm = None
        if rank == 0:
            valid_genomic_ds = GenomicAuxDataset(
                args.mlm_fasta, args.gff3, cache, args.h3k27ac_bigwig,
                args.atac_bigwig, args.conservation_bigwig,
                split="valid", context_length=CONTEXT_LENGTH,
                sample_count=args.validation_samples, seed=args.seed,
            )
            valid_genomic = DataLoader(
                valid_genomic_ds, batch_size=args.eval_batch_size, shuffle=False,
                num_workers=0, pin_memory=True,
            )
            valid_mlm_ds = BEDGenomeDataset(
                args.mlm_bed, args.mlm_fasta, CONTEXT_LENGTH,
                sequence_lengths=[CONTEXT_LENGTH], split="valid",
                max_samples=args.validation_samples, seed=args.seed,
            )
            from vericomp.data import SpanMLMCollator
            valid_mlm = DataLoader(
                valid_mlm_ds, batch_size=args.eval_batch_size, shuffle=False,
                collate_fn=SpanMLMCollator(), num_workers=0, pin_memory=True,
            )

        run_config = {
            "start_time_utc": datetime.now(timezone.utc).isoformat(),
            "task": ["MLM", "GENCODE", "H3K27ac", "ATAC", "phastCons"],
            "source_checkpoint": str(Path(args.checkpoint).resolve()),
            "loaded_backbone_tensors": loaded,
            "parameters_total": params,
            "world_size": world,
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "training_method": "joint shared-backbone multi-task update; MLM and all four supervised tasks every optimizer step",
            "context_length": CONTEXT_LENGTH,
            "supervised_target_length": TARGET_LENGTH,
            "supervised_target_center_offset": TARGET_OFFSET,
            "supervised_tasks": {
                "GENCODE": {
                    "path": str(Path(args.gff3).resolve()),
                    "labels": list(ANNOTATION_NAMES),
                    "loss": "per-class balanced binary cross entropy",
                    "split": "train excludes chr8, chr9, chr10; chr10 validation; chr8/chr9 test",
                },
                "H3K27ac": {
                    "path": str(Path(args.h3k27ac_bigwig).resolve()),
                    "target_transform": "log1p(signal p-value), clipped at 1000",
                    "loss": "SmoothL1 beta=0.5",
                    "biosample": "HUVEC",
                },
                "ATAC": {
                    "path": str(Path(args.atac_bigwig).resolve()),
                    "target_transform": "log1p(signal p-value), clipped at 1000",
                    "loss": "SmoothL1 beta=0.5",
                    "biosample": "adult female heart left ventricle",
                },
                "phastCons": {
                    "path": str(Path(args.conservation_bigwig).resolve()),
                    "target_transform": "native phastCons100way score clipped to [0, 1]; missing bins masked",
                    "loss": "masked SmoothL1 beta=0.5",
                    "assembly": "hg38",
                },
            },
            "mlm_fasta": str(Path(args.mlm_fasta).resolve()),
            "mlm_bed": str(Path(args.mlm_bed).resolve()),
            "mlm_split": "train",
            "mlm_masking": "15% span masking, mean span length 8, paired reverse-complement masks",
            "mlm_batch_per_gpu": args.mlm_batch_size,
            "genomic_batch_per_gpu": args.genomic_batch_size,
            "target_loss_ratio": args.target_loss_ratio,
            "dynamic_weighting": "each detached weighted supervised batch loss, including phastCons, equals 0.3 times global-mean MLM cross entropy; router regularization is separate",
            "router_auxiliary": "mean of MLM and genomic-branch router terms, not assigned to a supervised task",
            "validation": "32 fixed central windows on chr10 plus 32 MLM validation windows; best checkpoint minimizes composite validation objective",
            "test": "chr8/chr9 genomic held-out windows only after training",
            "steps": args.steps,
            "evaluation_interval": args.eval_every,
            "data_loading_profile_steps": args.profile_data_steps,
            "learning_rate": args.lr,
            "learning_rate_end": args.lr_end,
            "learning_rate_schedule": "linear",
            "weight_decay": args.weight_decay,
            "gradient_clip": args.max_grad_norm,
            "router_loss_coef": model.router_loss_coef,
            "precision": "BF16 autocast for model compute; task losses accumulated in FP32",
            "activation_checkpointing": args.activation_checkpointing,
            "seed": args.seed,
            "arguments": vars(args),
        }
        if rank == 0:
            print_resolved_config(
                entrypoint="pretrain.multitask_mlm_genomic", args=args,
                model_config=config,
                runtime={"world_size": world, "device": torch.cuda.get_device_name(device),
                         "loaded_backbone_tensors": loaded, "parameters_total": params},
                extra={"resolved_run_config": run_config},
            )
            write_run_config(
                output, entrypoint="pretrain.multitask_mlm_genomic",
                model_path=Path(__file__).resolve().parents[1] / "vericomp/modeling.py",
                model_class="MultiTaskGenomicModel", model_config=config,
                args=args, runtime=run_config,
                extra={"data": {"gff3": args.gff3, "h3k27ac": args.h3k27ac_bigwig,
                                "atac": args.atac_bigwig,
                                "conservation": args.conservation_bigwig,
                                "mlm_fasta": args.mlm_fasta,
                                "mlm_bed": args.mlm_bed}},
            )
            writer.log("tasks=MLM,GENCODE,H3K27ac,ATAC,phastCons; initializing from "
                       + str(Path(args.checkpoint).resolve()))
        if distributed:
            dist.barrier()

        if args.smoke_test:
            smoke_start = time.perf_counter()
            mlm_batch = _move(next(mlm_iter), device)
            genomic_batch = _move(next(genomic_iter), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                result = ddp(mlm_batch, genomic_batch)
            if not torch.isfinite(result["loss"]):
                raise RuntimeError("non-finite loss in genomic multi-task smoke test")
            result["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                ddp.parameters(), args.max_grad_norm
            )
            optimizer.step()
            ratios = {name: float(value.detach())
                      for name, value in result["task_ratios"].items()}
            if any(not math.isclose(value, args.target_loss_ratio,
                                    rel_tol=0.03, abs_tol=0.01)
                   for value in ratios.values()):
                raise RuntimeError(f"dynamic loss ratios out of tolerance: {ratios}")
            if rank == 0:
                writer.log(json.dumps({
                    "smoke_test": "one update completed",
                    "tasks": ["MLM", "GENCODE", "H3K27ac", "ATAC", "phastCons"],
                    "mlm_loss": float(result["mlm_loss"].detach()),
                    "task_losses": {
                        name: float(value.detach())
                        for name, value in result["task_losses"].items()
                    },
                    "dynamic_weights": {
                        name: float(value.detach())
                        for name, value in result["task_weights"].items()
                    },
                    "weighted_task_to_mlm_ratios": ratios,
                    "router_aux_loss": float(result["router_aux"].detach()),
                    "grad_norm": float(grad_norm.detach()),
                    "optimizer_update_elapsed_s": time.perf_counter() - smoke_start,
                    "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
                }, ensure_ascii=False))
            del result
            model.eval()
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                inference_output = model(mlm_batch, genomic_batch)
            if not torch.isfinite(inference_output["loss"]):
                raise RuntimeError("non-finite inference loss in smoke test")
            if rank == 0:
                writer.log(json.dumps({
                    "smoke_test": "inference pass completed",
                    "loss": float(inference_output["loss"].detach()),
                }, ensure_ascii=False))
            if distributed:
                dist.barrier()
            return

        best = math.inf
        history = []
        profile_samples = []
        start_time = time.perf_counter()
        mlm_epoch = 0
        for step in range(1, args.steps + 1):
            lr = args.lr + (args.lr_end - args.lr) * (
                (step - 1) / max(1, args.steps - 1)
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            profile_this_step = step <= args.profile_data_steps
            if profile_this_step:
                torch.cuda.synchronize(device)
                data_start = time.perf_counter()
            try:
                mlm_batch = next(mlm_iter)
            except StopIteration:
                mlm_epoch += 1
                mlm_sampler.set_epoch(mlm_epoch)
                mlm_iter = iter(mlm_loader)
                mlm_batch = next(mlm_iter)
            try:
                genomic_batch = next(genomic_iter)
            except StopIteration:
                genomic_iter = iter(genomic_loader)
                genomic_batch = next(genomic_iter)
            mlm_batch = _move(mlm_batch, device)
            genomic_batch = _move(genomic_batch, device)
            if profile_this_step:
                torch.cuda.synchronize(device)
                data_wait_s = time.perf_counter() - data_start
                compute_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                result = ddp(mlm_batch, genomic_batch)
            if not torch.isfinite(result["loss"]):
                raise RuntimeError(f"non-finite total loss at step={step}")
            result["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                ddp.parameters(), args.max_grad_norm
            )
            optimizer.step()
            if profile_this_step:
                torch.cuda.synchronize(device)
                compute_s = time.perf_counter() - compute_start
                timing = torch.tensor(
                    [data_wait_s, compute_s], device=device, dtype=torch.float64
                )
                if distributed:
                    dist.all_reduce(timing, op=dist.ReduceOp.MAX)
                data_wait_s, compute_s = map(float, timing.tolist())
                profile_samples.append((data_wait_s, compute_s))
            if step == 1 or step % 50 == 0:
                accuracy, masked = masked_accuracy(
                    result["mlm"]["logits"].detach(), mlm_batch["labels"]
                )
                values = [
                    float(result["loss"].detach()),
                    float(result["mlm_loss"].detach()),
                    float(result["router_aux"].detach()),
                    *[float(result["task_losses"][n].detach()) for n in model.TASKS],
                    *[float(result["task_weights"][n].detach()) for n in model.TASKS],
                    *[float(result["task_ratios"][n].detach()) for n in model.TASKS],
                ]
                reduced = _reduce_values(values, device, average=True)
                correct, mask_count = _reduce_values(
                    [accuracy, masked], device, average=False
                )
                record = {
                    "step": step,
                    "tasks": ["MLM", "GENCODE", "H3K27ac", "ATAC", "phastCons"],
                    "train_loss": reduced[0],
                    "mlm_loss": reduced[1],
                    "router_aux_loss": reduced[2],
                    "gencode_loss": reduced[3],
                    "h3k27ac_loss": reduced[4],
                    "atac_loss": reduced[5],
                    "conservation_loss": reduced[6],
                    "gencode_dynamic_weight": reduced[7],
                    "h3k27ac_dynamic_weight": reduced[8],
                    "atac_dynamic_weight": reduced[9],
                    "conservation_dynamic_weight": reduced[10],
                    "gencode_to_mlm_ratio": reduced[11],
                    "h3k27ac_to_mlm_ratio": reduced[12],
                    "atac_to_mlm_ratio": reduced[13],
                    "conservation_to_mlm_ratio": reduced[14],
                    "mlm_accuracy": correct / max(1.0, mask_count),
                    "masked_tokens_global": mask_count,
                    "lr": lr,
                    "grad_norm": float(grad_norm.detach()),
                    "elapsed_s": time.perf_counter() - start_time,
                    "gpu_memory_peak_bytes": torch.cuda.max_memory_allocated(device),
                }
                if profile_samples:
                    mean_data_wait, mean_compute = np.asarray(
                        profile_samples, dtype=np.float64
                    ).mean(axis=0)
                    record.update({
                        "profiled_updates": len(profile_samples),
                        "data_wait_s_mean": float(mean_data_wait),
                        "compute_s_mean": float(mean_compute),
                        "data_wait_fraction": float(
                            mean_data_wait / max(1e-12, mean_data_wait + mean_compute)
                        ),
                    })
                    profile_samples.clear()
                if rank == 0:
                    writer.log(json.dumps(record, ensure_ascii=False))

            if step % args.eval_every == 0 or step == args.steps:
                if distributed:
                    dist.barrier()
                if rank == 0:
                    random.seed(args.seed + step)
                    valid = _evaluate(model, valid_mlm, valid_genomic, device)
                    improved = valid["composite_objective"] < best
                    if improved:
                        best = valid["composite_objective"]
                    valid_record = {"step": step, **valid}
                    history.append(valid_record)
                    writer.log(json.dumps({"step": step, "valid": valid}, ensure_ascii=False))
                    _save_checkpoint(model, output, args, step, valid_record, best,
                                     best=improved)
                    _atomic_json(
                        {"best_validation_objective": best, "history": history},
                        output / "metrics.json",
                    )
                if distributed:
                    dist.barrier()

        if distributed:
            dist.barrier()
        if rank == 0 and (output / "best.pt").is_file():
            best_state = torch.load(output / "best.pt", map_location="cpu",
                                    weights_only=False)
            model.load_state_dict(best_state["model"], strict=True)
            test_ds = GenomicAuxDataset(
                args.mlm_fasta, args.gff3, cache, args.h3k27ac_bigwig,
                args.atac_bigwig, args.conservation_bigwig,
                split="test", context_length=CONTEXT_LENGTH,
                sample_count=args.test_samples, seed=args.seed,
            )
            test_loader = DataLoader(
                test_ds, batch_size=args.eval_batch_size, shuffle=False,
                num_workers=0, pin_memory=True,
            )
            # Test evaluation needs matching MLM batches; the MLM test split is
            # used only for reporting, never for checkpoint selection.
            test_mlm_ds = BEDGenomeDataset(
                args.mlm_bed, args.mlm_fasta, CONTEXT_LENGTH,
                sequence_lengths=[CONTEXT_LENGTH], split="test",
                max_samples=args.test_samples, seed=args.seed,
            )
            from vericomp.data import SpanMLMCollator
            test_mlm_loader = DataLoader(
                test_mlm_ds, batch_size=args.eval_batch_size, shuffle=False,
                collate_fn=SpanMLMCollator(), num_workers=0, pin_memory=True,
            )
            test_metrics = _evaluate(model, test_mlm_loader, test_loader, device)
            _atomic_json({"checkpoint": "best.pt", **test_metrics},
                         output / "test_metrics.json")
            writer.log(json.dumps({"test": test_metrics}, ensure_ascii=False))
        if distributed:
            dist.barrier()
    finally:
        writer.close()
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
