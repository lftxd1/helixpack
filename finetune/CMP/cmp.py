"""VeriComp DNALongBench CMP, aligned with the current downstream protocol.

The current protocol precomputes hidden states from 131k-bp trunks,
coverage-pools them into 448 contact-map bins, and fine-tunes only a 2D CNN
contact-map head.  This implementation follows that protocol while using
VeriComp's decoder feature as the ``before_lm_head`` representation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from finetune.eQTL.eqtl import load_pretrained_backbone
from finetune.run_config import (print_resolved_config, resolve_model_path,
                                  write_finetune_config)
from vericomp import VeriCompConfig
from vericomp.model_presets import available_model_presets, model_config_from_checkpoint
from vericomp.backbone import build_backbone, extract_feature
from vericomp.cmp_data import CMP_SUBSETS, load_cmp_splits, normalize_cmp_subset


# GeneZip official CMP settings.
NUM_BINS = 448
BIN_SIZE = 2048
UPPER_OFFSET = 2
MAX_COVERED_BP = NUM_BINS * BIN_SIZE
HEAD_DIM = 128
DIST_EMB_DIM = 16
CNN1D_CHANNELS = 128
CNN1D_LAYERS = 2
CNN1D_KERNEL = 5
CNN2D_CHANNELS = (64, 64, 64)
CNN2D_KERNEL = 3
DROPOUT = 0.1


def upper_indices(device: torch.device) -> torch.Tensor:
    return torch.triu_indices(NUM_BINS, NUM_BINS, offset=UPPER_OFFSET, device=device)


def build_strata() -> list[np.ndarray]:
    indices = torch.triu_indices(NUM_BINS, NUM_BINS, offset=UPPER_OFFSET)
    distances = (indices[1] - indices[0]).numpy()
    return [np.flatnonzero(distances == distance) for distance in range(UPPER_OFFSET, NUM_BINS)]


STRATA = build_strata()


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return float("nan")
    x = x[valid].astype(np.float64, copy=False)
    y = y[valid].astype(np.float64, copy=False)
    x = x - x.mean()
    y = y - y.mean()
    denominator = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / denominator) if denominator > 0 else float("nan")


def _scc_from_upper(pred: np.ndarray, target: np.ndarray) -> float:
    """GeneZip SCC: distance-stratified Pearson, weighted by valid counts."""
    weighted_sum = 0.0
    weight_total = 0
    for indices in STRATA:
        if indices.size == 0:
            continue
        value_pred = pred[indices]
        value_target = target[indices]
        valid = np.isfinite(value_pred) & np.isfinite(value_target)
        if valid.sum() < 2:
            continue
        value_pred = value_pred[valid].astype(np.float64, copy=False)
        value_target = value_target[valid].astype(np.float64, copy=False)
        value_pred -= value_pred.mean()
        value_target -= value_target.mean()
        denominator = math.sqrt(float(np.dot(value_pred, value_pred) * np.dot(value_target, value_target)))
        if denominator <= 0:
            continue
        corr = float(np.dot(value_pred, value_target) / denominator)
        count = int(valid.sum())
        weighted_sum += corr * count
        weight_total += count
    return float(weighted_sum / weight_total) if weight_total else float("nan")


def _upper_pearson(pred: np.ndarray, target: np.ndarray) -> float:
    return _pearson(pred.reshape(-1), target.reshape(-1))


def _masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(pred) & torch.isfinite(target)
    if not bool(valid.any()):
        return pred.sum() * 0.0
    return (pred[valid] - target[valid]).square().mean()


class CMPContactHead(nn.Module):
    """GeneZip's 2D-CNN contact-map head."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, HEAD_DIM)
        cnn1d: list[nn.Module] = []
        for layer_index in range(CNN1D_LAYERS):
            in_channels = HEAD_DIM if layer_index == 0 else CNN1D_CHANNELS
            cnn1d.extend((
                nn.Conv1d(in_channels, CNN1D_CHANNELS, kernel_size=CNN1D_KERNEL,
                          padding=CNN1D_KERNEL // 2),
                nn.GELU(),
                nn.Dropout(DROPOUT),
            ))
        self.cnn1d = nn.Sequential(*cnn1d)
        self.dist_embedding = nn.Embedding(NUM_BINS, DIST_EMB_DIM)
        distance = torch.arange(NUM_BINS)
        self.register_buffer("distance_index", (distance[None, :] - distance[:, None]).abs(), persistent=False)

        pair_dim = HEAD_DIM * 3 + DIST_EMB_DIM
        conv2d: list[nn.Module] = []
        in_channels = pair_dim
        for out_channels in CNN2D_CHANNELS:
            conv2d.extend((
                nn.Conv2d(in_channels, out_channels, kernel_size=CNN2D_KERNEL,
                          padding=CNN2D_KERNEL // 2),
                nn.GELU(),
                nn.Dropout2d(DROPOUT),
            ))
            in_channels = out_channels
        conv2d.append(nn.Conv2d(in_channels, 1, kernel_size=1))
        self.cnn2d = nn.Sequential(*conv2d)

    def forward(self, h_bins: torch.Tensor) -> torch.Tensor:
        hidden = self.proj(h_bins)
        hidden = self.cnn1d(hidden.transpose(1, 2)).transpose(1, 2)
        batch, bins, dim = hidden.shape
        hidden_i = hidden.unsqueeze(2).expand(batch, bins, bins, dim)
        hidden_j = hidden.unsqueeze(1).expand(batch, bins, bins, dim)
        distance = self.dist_embedding(self.distance_index.to(hidden.device))
        pair = torch.cat((hidden_i, hidden_j, hidden_i * hidden_j,
                          distance.unsqueeze(0).expand(batch, -1, -1, -1)), dim=-1)
        logits = self.cnn2d(pair.permute(0, 3, 1, 2)).squeeze(1)
        return 0.5 * (logits + logits.transpose(-1, -2))


def _cache_name(sample_id: str) -> str:
    return hashlib.sha1(str(sample_id).encode("utf-8")).hexdigest() + ".pt"


class CMPHbinsDataset(Dataset):
    def __init__(self, source, cache_dir: Path, target_index: int,
                 max_samples: int | None = None, preload_cache: bool = False,
                 packed_cache: dict | None = None,
                 target_cache: torch.Tensor | None = None):
        self.source = source
        # The raw CMP rows contain a 1-Mb sequence.  It is needed only during
        # h-bin precomputation; loading it again for every cached training
        # example needlessly stalls the CPU data path.
        metadata_columns = ("sample_id",) if target_cache is not None else ("sample_id", "label_ut")
        self.metadata = source.split.select_columns(metadata_columns)
        self.cache_dir = Path(cache_dir)
        self.target_index = int(target_index)
        self.length = len(source) if not max_samples or max_samples <= 0 else min(len(source), int(max_samples))
        self.preload_cache = bool(preload_cache)
        self._h_bins = None
        self._targets = None
        self._packed_h_bins = None
        self._packed_indices = None
        self._target_cache = target_cache
        if packed_cache is not None:
            started = time.perf_counter()
            sample_metadata = self.source.split.select_columns(["sample_id"])
            metadata_rows = sample_metadata[:self.length]
            sample_ids = [str(value) for value in metadata_rows["sample_id"]]
            packed_index = packed_cache["index"]
            self._packed_h_bins = packed_cache["h_bins"]
            self._packed_indices = torch.tensor(
                [packed_index[_cache_name(sample_id)] for sample_id in sample_ids],
                dtype=torch.long,
            )
            del metadata_rows, sample_ids
            self.metadata = sample_metadata
            print(json.dumps({
                "phase": "packed_cache_indexed",
                "cache_dir": str(self.cache_dir.resolve()),
                "examples": self.length,
                "seconds": round(time.perf_counter() - started, 2),
            }), flush=True)
        elif self.preload_cache:
            started = time.perf_counter()
            metadata_rows = self.metadata[:self.length]
            sample_ids = [str(value) for value in metadata_rows["sample_id"]]
            label_column = self.source.split.data.column("label_ut").slice(0, self.length)
            target_rows = []
            for chunk in label_column.iterchunks():
                values = chunk.to_numpy(zero_copy_only=False)
                target_rows.append(torch.from_numpy(
                    np.asarray(values[:, :, self.target_index], dtype=np.float32).copy()))
            self._targets = torch.cat(target_rows, dim=0)
            del target_rows, label_column
            if packed_cache is not None:
                packed_index = packed_cache["index"]
                packed_indices = [packed_index[_cache_name(sample_id)] for sample_id in sample_ids]
                self._packed_h_bins = packed_cache["h_bins"]
                self._packed_indices = torch.tensor(packed_indices, dtype=torch.long)
            else:
                h_bins_rows = []
                for sample_id in sample_ids:
                    payload = torch.load(self.cache_dir / _cache_name(sample_id),
                                         map_location="cpu", weights_only=True)
                    h_bins_rows.append(payload["h_bins"].contiguous())
                self._h_bins = torch.stack(h_bins_rows)
                del h_bins_rows
            del metadata_rows, sample_ids
            self.source = None
            self.metadata = None
            print(json.dumps({
                "phase": "cache_loaded_to_ram",
                "cache_dir": str(self.cache_dir.resolve()),
                "examples": self.length,
                "storage": "packed" if packed_cache is not None else "per_file",
                "h_bins_shape": list(self._packed_h_bins.shape) if packed_cache is not None else list(self._h_bins.shape),
                "h_bins_gib": round(
                    (self._packed_h_bins if packed_cache is not None else self._h_bins).numel()
                    * (self._packed_h_bins if packed_cache is not None else self._h_bins).element_size() / 2**30, 3),
                "seconds": round(time.perf_counter() - started, 2),
            }), flush=True)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        if self._packed_h_bins is not None:
            row = self._packed_indices[int(index)]
            target = (self._target_cache[int(index)] if self._target_cache is not None else
                      torch.as_tensor(self.source.split[int(index)]["label_ut"], dtype=torch.float32)[:, self.target_index])
            return {
                "h_bins": self._packed_h_bins[row],
                "target": target,
            }
        if self.preload_cache:
            row = self._packed_indices[int(index)] if self._packed_indices is not None else int(index)
            h_bins = self._packed_h_bins[row] if self._packed_h_bins is not None else self._h_bins[row]
            return {"h_bins": h_bins, "target": self._targets[int(index)]}
        item = self.metadata[int(index)]
        sample_id = str(item["sample_id"])
        payload = torch.load(self.cache_dir / _cache_name(sample_id), map_location="cpu", weights_only=True)
        return {
            "h_bins": payload["h_bins"],
            "target": torch.as_tensor(item["label_ut"], dtype=torch.float32)[:, self.target_index],
        }


class VeriCompCMPModel(nn.Module):
    def __init__(self, config: VeriCompConfig, *, model_name: str = "vericomp",
                 feature: str = "decoder", trunk_length: int = 12_800):
        super().__init__()
        self.model_name = model_name
        self.feature = feature
        self.trunk_length = int(trunk_length)
        self.backbone = build_backbone(model_name, config)
        self.head = CMPContactHead(config.d_model)

    @torch.inference_mode()
    def encode_hbins(self, input_ids: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, float]:
        """Encode one sequence; kept as a compatibility wrapper."""
        if input_ids.dim() != 1:
            raise ValueError(f"expected one sequence [L], got {tuple(input_ids.shape)}")
        h_bins, router = self.encode_hbins_batch(input_ids.unsqueeze(0), device)
        return h_bins[0], router

    @torch.inference_mode()
    def encode_hbins_batch(self, input_ids: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, float]:
        """Encode a batch of equal-length sequences without autograd."""
        if input_ids.dim() != 2:
            raise ValueError(f"expected a batch [B,L], got {tuple(input_ids.shape)}")
        batch_size, sequence_length = input_ids.shape
        input_ids = input_ids.to(device=device, dtype=torch.long)
        numerator = torch.zeros((batch_size, NUM_BINS, self.head.proj.in_features),
                                device=device, dtype=torch.float32)
        denominator = torch.zeros((batch_size, NUM_BINS), device=device, dtype=torch.float32)
        router_values = []
        self.backbone.eval()
        for start in range(0, int(sequence_length), self.trunk_length):
            stop = min(start + self.trunk_length, int(sequence_length))
            chunk = input_ids[:, start:stop]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                features, output = extract_feature(self.backbone, chunk, self.feature)
            if features.shape[1] != stop - start:
                features = F.interpolate(features.transpose(1, 2), size=stop - start,
                                         mode="linear", align_corners=False).transpose(1, 2)
            router_values.append(float(output["loss_router"].float().item()))

            clip_left = max(0, -start)
            clip_right = min(stop - start, MAX_COVERED_BP - start)
            if clip_right <= clip_left:
                del features, output, chunk
                continue
            for bin_index in range(NUM_BINS):
                bin_start = bin_index * BIN_SIZE
                bin_stop = bin_start + BIN_SIZE
                left = max(start, bin_start)
                right = min(start + clip_right, bin_stop)
                if right > left:
                    local_left = left - start
                    local_right = right - start
                    numerator[:, bin_index] += features[:, local_left:local_right].float().sum(dim=1)
                    denominator[:, bin_index] += float(right - left)
            # The model output keeps a reference to decoder_feature.  Release
            # both objects before launching the next 128k trunk so batched
            # inference does not accumulate activations across trunks.
            del features, output, chunk
        h_bins = numerator / denominator.clamp_min(1.0).unsqueeze(-1)
        return h_bins.to(dtype=torch.bfloat16).cpu(), float(np.mean(router_values)) if router_values else 0.0

    def forward(self, h_bins: torch.Tensor, target: torch.Tensor | None = None):
        prediction = self.head(h_bins)
        upper = upper_indices(prediction.device)
        prediction_upper = prediction[:, upper[0], upper[1]]
        result = {"pred": prediction_upper}
        if target is not None:
            result["loss"] = _masked_mse(prediction_upper, target)
        return result


def _write_hbin_cache(path: Path, h_bin: torch.Tensor, router: float):
    """Write one detached h-bin tensor without retaining batch storage."""
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    torch.save({"h_bins": h_bin.clone().contiguous(), "router": router}, tmp_path)
    os.replace(tmp_path, path)


def precompute_hbins(model: VeriCompCMPModel, datasets, cache_dir: Path, device: torch.device,
                     batch_size: int = 1, rank: int = 0, world_size: int = 1,
                     writer_workers: int = 4, writer_queue_batches: int = 16):
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_tasks = [(dataset, index) for dataset in datasets for index in range(len(dataset))]
    total = len(all_tasks)
    rank = int(rank)
    world_size = max(1, int(world_size))
    if rank < 0 or rank >= world_size:
        raise ValueError(f"invalid precompute rank={rank} for world_size={world_size}")
    tasks = all_tasks[rank::world_size]
    local_total = len(tasks)
    completed = 0
    batch_size = max(1, int(batch_size))
    writer_workers = max(1, int(writer_workers))
    max_pending = batch_size * max(1, int(writer_queue_batches))
    pending = set()
    with ThreadPoolExecutor(max_workers=writer_workers,
                            thread_name_prefix=f"cmp-cache-writer-{rank}") as writer:
        for batch_start in range(0, local_total, batch_size):
            task_batch = tasks[batch_start:batch_start + batch_size]
            items = [dataset[index] for dataset, index in task_batch]
            missing = []
            for item in items:
                path = cache_dir / _cache_name(item["sample_id"])
                if not path.exists():
                    missing.append((item, path))
            if missing:
                lengths = {int(item["input_ids"].numel()) for item, _ in missing}
                if len(lengths) != 1:
                    raise ValueError(f"batched CMP precompute requires equal sequence lengths, got {lengths}")
                input_ids = torch.stack([item["input_ids"] for item, _ in missing])
                h_bins, router = model.encode_hbins_batch(input_ids, device)
                for row, (_, path) in enumerate(missing):
                    pending.add(writer.submit(_write_hbin_cache, path,
                                              h_bins[row].clone().contiguous(), router))
            if len(pending) >= max_pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
            completed += len(items)
            if completed == len(items) or completed % 50 < len(items) or completed == local_total:
                print(json.dumps({"phase": "precompute_hbins", "completed": completed,
                                  "local_total": local_total, "global_total": total,
                                  "rank": rank, "world_size": world_size,
                                  "cache_dir": str(cache_dir), "batch_size": batch_size,
                                  "writer_workers": writer_workers,
                                  "writer_queue_batches": int(writer_queue_batches)}), flush=True)
        for future in pending:
            future.result()


def evaluate(model, loader, device):
    model.eval()
    losses, scc_values, corr_values = [], [], []
    with torch.inference_mode():
        for batch in loader:
            h_bins = batch["h_bins"].to(device=device, dtype=torch.bfloat16, non_blocking=True)
            target = batch["target"].to(device=device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(h_bins, target)
            losses.append(float(output["loss"].float().item()))
            prediction = output["pred"].float().cpu().numpy()
            truth = target.float().cpu().numpy()
            for index in range(prediction.shape[0]):
                scc_values.append(_scc_from_upper(prediction[index], truth[index]))
                corr_values.append(_upper_pearson(prediction[index], truth[index]))
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "scc": float(np.nanmean(scc_values)) if scc_values else float("nan"),
        "corr": float(np.nanmean(corr_values)) if corr_values else float("nan"),
        "examples": len(scc_values),
    }


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-size", choices=("auto", *available_model_presets()),
                        default="auto",
                        help="architecture preset; auto reads new checkpoint metadata")
    parser.add_argument("--model", default="vericomp")
    parser.add_argument("--feature", choices=("backbone", "decoder", "base"), default="decoder")
    parser.add_argument("--subset", choices=tuple(CMP_SUBSETS), default="HFF")
    parser.add_argument("--data-root", default="./data/dnalongbench")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=20000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-valid-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--trunk-length", type=int, default=131_072)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4,
                        help="batches prefetched by each training DataLoader worker")
    parser.add_argument("--preload-cache", action="store_true",
                        help="load all cached h-bins into host RAM before training")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--precompute-batch-size", type=int, default=1)
    parser.add_argument("--precompute-rank", type=int, default=0)
    parser.add_argument("--precompute-world-size", type=int, default=1)
    parser.add_argument("--writer-workers", type=int, default=4)
    parser.add_argument("--writer-queue-batches", type=int, default=16)
    parser.add_argument("--precompute-only", action="store_true",
                        help="precompute h-bin features and exit without head training")
    parser.add_argument("--skip-precompute", action="store_true",
                        help="reuse an already complete h-bin cache without scanning raw sequences")
    parser.add_argument("--output-dir", default=str(Path("weights/finetune/CMP") / time.strftime("%Y%m%d")))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CMP downstream evaluation requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.subset = normalize_cmp_subset(args.subset)

    checkpoint_tag = Path(args.checkpoint).stem
    cache_dir = Path(args.cache_dir) if args.cache_dir else (
        Path(args.data_root) / "cmp_cache" / f"{checkpoint_tag}-{args.model.replace('/', '_')}-"
        f"{args.feature}-trunk{args.trunk_length}"
    )
    target_paths = {
        split_name: cache_dir / f"targets_{args.subset}_{split_name}.pt"
        for split_name in ("train", "valid", "test")
    }
    target_cache_ready = all(path.is_file() for path in target_paths.values())
    raw_columns = None
    if args.skip_precompute:
        raw_columns = ("sample_id",) if target_cache_ready else ("sample_id", "label_ut")
    train_raw, valid_raw, test_raw = load_cmp_splits(
        args.data_root, args.subset, args.max_train_samples,
        args.max_valid_samples, args.max_test_samples, columns=raw_columns,
    )
    config = model_config_from_checkpoint(
        args.checkpoint, args.model_size, force_flash_attention=True,
        activation_checkpointing=False,
    )
    model = VeriCompCMPModel(config, model_name=args.model, feature=args.feature,
                            trunk_length=args.trunk_length).to(device)
    loaded, skipped = load_pretrained_backbone(model, args.checkpoint)

    if args.skip_precompute:
        print(json.dumps({"phase": "precompute_skipped", "cache_dir": str(cache_dir.resolve()),
                          "reason": "reuse_complete_cache"}), flush=True)
    else:
        precompute_hbins(model, (train_raw, valid_raw, test_raw), cache_dir, device,
                         batch_size=args.precompute_batch_size,
                         rank=args.precompute_rank,
                         world_size=args.precompute_world_size,
                         writer_workers=args.writer_workers,
                         writer_queue_batches=args.writer_queue_batches)
    if args.precompute_only:
        print(json.dumps({
            "phase": "precompute_complete",
            "cache_dir": str(cache_dir.resolve()),
            "trunk_length": args.trunk_length,
            "feature": args.feature,
            "inference_mode": True,
            "train_examples": len(train_raw),
            "valid_examples": len(valid_raw),
            "test_examples": len(test_raw),
        }, indent=2), flush=True)
        return
    packed_cache = None
    packed_path = cache_dir / "h_bins_packed.pt"
    if args.preload_cache and packed_path.is_file():
        started = time.perf_counter()
        packed_cache = torch.load(packed_path, map_location="cpu", weights_only=True)
        print(json.dumps({
            "phase": "packed_cache_loaded_to_ram",
            "path": str(packed_path),
            "shape": list(packed_cache["h_bins"].shape),
            "gib": round(packed_cache["h_bins"].numel() * packed_cache["h_bins"].element_size() / 2**30, 3),
            "seconds": round(time.perf_counter() - started, 2),
        }), flush=True)
    target_cache = {}
    if args.preload_cache and target_cache_ready:
        started = time.perf_counter()
        for split_name, path in target_paths.items():
            target_cache[split_name] = torch.load(path, map_location="cpu", weights_only=True)
        print(json.dumps({
            "phase": "target_cache_loaded_to_ram",
            "subset": args.subset,
            "gib": round(sum(value.numel() * value.element_size() for value in target_cache.values()) / 2**30, 3),
            "seconds": round(time.perf_counter() - started, 2),
        }), flush=True)
    train_ds = CMPHbinsDataset(train_raw, cache_dir, CMP_SUBSETS[args.subset],
                                preload_cache=args.preload_cache, packed_cache=packed_cache,
                                target_cache=target_cache.get("train"))
    valid_ds = CMPHbinsDataset(valid_raw, cache_dir, CMP_SUBSETS[args.subset],
                                preload_cache=args.preload_cache, packed_cache=packed_cache,
                                target_cache=target_cache.get("valid"))
    test_ds = CMPHbinsDataset(test_raw, cache_dir, CMP_SUBSETS[args.subset],
                               preload_cache=args.preload_cache, packed_cache=packed_cache,
                               target_cache=target_cache.get("test"))
    loader_options = {"pin_memory": True}
    if args.num_workers > 0:
        loader_options.update({
            "num_workers": args.num_workers,
            "prefetch_factor": max(1, args.prefetch_factor),
            "persistent_workers": True,
        })
    else:
        loader_options["num_workers"] = 0
    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_options),
        "valid": DataLoader(valid_ds, batch_size=args.eval_batch_size, shuffle=False, **loader_options),
        "test": DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, **loader_options),
    }

    # GeneZip precomputes the encoder output and optimizes only the contact head.
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    params = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.weight_decay,
        fused=(device.type == "cuda"),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_finetune_config(
        output_dir, task="CMP", subtask=args.subset, args=args,
        model_path=resolve_model_path(args.model),
        model_config=config,
        extra={"data_root": str(Path(args.data_root).resolve()), "cache_dir": str(cache_dir.resolve()),
               "protocol": "GeneZip CMP: 12.8k trunks, h_bins, 2dcnn head, weighted SCC",
               "preload_cache": args.preload_cache, "num_workers": args.num_workers,
               "prefetch_factor": args.prefetch_factor},
    )
    print_resolved_config(
        task="CMP", subtask=args.subset, args=args,
        model_path=resolve_model_path(args.model), model_config=config,
        metadata={
            "checkpoint": {
                "path": str(Path(args.checkpoint).expanduser().resolve()),
                "loaded_backbone_tensors": loaded,
                "skipped_backbone_tensors": skipped,
            },
            "data": {
                "subset": args.subset,
                "target_index": CMP_SUBSETS[args.subset],
                "train_examples": len(train_ds),
                "valid_examples": len(valid_ds),
                "test_examples": len(test_ds),
                "num_workers": args.num_workers,
                "cache_dir": str(cache_dir.resolve()),
            },
            "protocol": {
                "num_bins": NUM_BINS,
                "bin_size": BIN_SIZE,
                "upper_offset": UPPER_OFFSET,
                "head": "2dcnn",
                "pooling": "coverage_aware_h_bins",
                "trunk_length": args.trunk_length,
                "encoder_frozen": True,
            },
            "optimization": {
                "optimizer": "AdamW",
                "train_batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size,
                "effective_batch_size": args.batch_size * args.grad_accumulation,
            },
            "parameters": {
                "total": sum(p.numel() for p in model.parameters()),
                "trainable": sum(p.numel() for p in params),
            },
            "runtime": {
                "device": str(device),
                "torch_version": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "autocast": "bfloat16" if device.type == "cuda" else "disabled",
            },
        },
    )

    best_scc = -math.inf
    history = []
    iterator = iter(loaders["train"])
    optimizer.zero_grad(set_to_none=True)
    start_time = time.perf_counter()
    for step in range(1, args.train_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loaders["train"])
            batch = next(iterator)
        h_bins = batch["h_bins"].to(device=device, dtype=torch.bfloat16, non_blocking=True)
        target = batch["target"].to(device=device, non_blocking=True)
        model.backbone.eval()
        model.head.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(h_bins, target)
            loss = output["loss"] / args.grad_accumulation
        loss.backward()
        if step % args.grad_accumulation == 0 or step == args.train_steps:
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 50 == 0 or step == args.train_steps:
            elapsed = max(time.perf_counter() - start_time, 1e-6)
            print(json.dumps({"step": step, "train_loss": float(output["loss"].detach().float()),
                              "step_s": elapsed / step}), flush=True)
        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.train_steps):
            valid = evaluate(model, loaders["valid"], device)
            test = evaluate(model, loaders["test"], device)
            record = {"step": step, "valid": valid, "test": test}
            history.append(record)
            print(json.dumps(record), flush=True)
            if np.isfinite(valid["scc"]) and valid["scc"] > best_scc:
                best_scc = valid["scc"]
                torch.save({"model": model.state_dict(), "step": step,
                            "valid": valid, "args": vars(args)}, output_dir / "best.pt")

    result = {"task": "CMP", "subtask": args.subset, "valid_best_scc": best_scc, "history": history}
    torch.save({"model": model.state_dict(), "result": result, "args": vars(args)}, output_dir / "last.pt")
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "valid_best_scc": best_scc}), flush=True)


if __name__ == "__main__":
    main()
