"""VeriComp downstream eQTL classification on DNALongBench.

Unlike the pretraining objective, this script pools the Transformer/MoE
bottleneck output ``[B, N, d_model]`` over ``N``.  It never pools the
full-resolution decoder or residual-skip output.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from vericomp import VeriCompConfig
from vericomp.backbone import build_backbone, extract_feature
from vericomp.eqtl_data import EQTLDataset
from vericomp.model_presets import (
    architecture_mismatches,
    available_model_presets,
    model_config_from_checkpoint,
)
from finetune.run_config import (print_resolved_config, resolve_model_path,
                                  write_finetune_config)


def auroc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    y, s = labels[order], scores[order]
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = np.empty_like(s, dtype=np.float64)
    i = 0
    rank = 1
    while i < len(s):
        j = i + 1
        while j < len(s) and s[j] == s[i]:
            j += 1
        ranks[i:j] = 0.5 * (rank + rank + j - i - 1)
        rank += j - i
        i = j
    pos_rank_sum = ranks[y == 1].sum()
    return float((pos_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def auprc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int((labels == 1).sum())
    if positives == 0 or labels.size == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    return float(precision[y == 1].sum() / positives)


N_TOKEN_ID = 5  # matches _BASE_TO_ID.get(base, 5) in eqtl_data


def _masked_mean_pool(features, output, input_ids, backbone):
    """Mean-pool bottleneck tokens while ignoring slots whose aggregated
    input mass is majority N padding.  The compressor's last cumulative
    allocation is replayed through its own _three_neighbours, so the N
    fraction per slot uses exactly the weights the model aggregated with."""
    compressor = getattr(backbone, "compressor", None)
    cumulative = getattr(compressor, "last_cumulative", None) if compressor is not None else None
    if cumulative is None:
        raise RuntimeError("masked pooling requires the compressor to expose last_cumulative")
    n_slots = features.shape[1]
    indices, weights = type(compressor)._three_neighbours(cumulative, n_slots)
    is_n = input_ids.eq(N_TOKEN_ID).to(torch.float32)
    n_mass = torch.zeros(features.shape[0], n_slots, device=features.device,
                         dtype=torch.float32)
    total_mass = torch.zeros_like(n_mass)
    for j in range(indices.shape[1]):
        n_mass.scatter_add_(1, indices[:, j], weights[:, j] * is_n)
        total_mass.scatter_add_(1, indices[:, j], weights[:, j])
    keep = (n_mass <= 0.5 * total_mass.clamp_min(1e-6)).to(features.dtype).unsqueeze(-1)
    total = keep.sum(dim=1).clamp_min(1.0)
    return ((features.float() * keep).sum(dim=1) / total).to(features.dtype)


class VeriCompEQTLClassifier(nn.Module):
    def __init__(self, config: VeriCompConfig, *, freeze_encoder: bool,
                 model_name: str = "vericomp", feature: str = "backbone",
                 dual_strand: bool = False, masked_pooling: bool = False,
                 quota_mask_n: bool = False, loss_cap: float | None = None):
        super().__init__()
        self.model_name = model_name
        self.feature = feature
        self.dual_strand = bool(dual_strand)
        self.masked_pooling = bool(masked_pooling)
        self.loss_cap = loss_cap
        self.backbone = build_backbone(model_name, config)
        if quota_mask_n:
            self.backbone.compressor.quota_mask_token_id = N_TOKEN_ID
        self.classifier = nn.Linear(config.d_model, 2)
        self.freeze_encoder = bool(freeze_encoder)
        if self.freeze_encoder:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        if self.freeze_encoder:
            self.backbone.eval()
            with torch.no_grad():
                features, output = extract_feature(
                    self.backbone, input_ids, self.feature,
                    dual_strand=self.dual_strand)
        else:
            features, output = extract_feature(
                self.backbone, input_ids, self.feature,
                dual_strand=self.dual_strand)

        if self.masked_pooling:
            pooled = _masked_mean_pool(features, output, input_ids, self.backbone)
        else:
            pooled = features.mean(dim=1)
        logits = self.classifier(pooled)
        result = {"logits": logits, "pooled": pooled}
        if labels is not None:
            if self.loss_cap is not None:
                per_sample = nn.functional.cross_entropy(logits, labels, reduction="none")
                overflow = (per_sample - self.loss_cap).clamp_min(0)
                result["loss"] = (per_sample.clamp(max=self.loss_cap) + 0.1 * overflow).mean()
            else:
                result["loss"] = nn.functional.cross_entropy(logits, labels)
        return result


def load_pretrained_backbone(model: VeriCompEQTLClassifier, path: str) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no state dict: {path}")

    checkpoint_config = checkpoint.get("model_config") if isinstance(checkpoint, dict) else None
    current_config = getattr(model.backbone, "config", None)
    if isinstance(checkpoint_config, dict) and current_config is not None:
        mismatches = architecture_mismatches(checkpoint_config, current_config)
        if mismatches:
            details = ", ".join(
                f"{name}={checkpoint_value!r} (current {current_value!r})"
                for name, (checkpoint_value, current_value) in mismatches.items()
            )
            raise ValueError(
                "checkpoint architecture does not match the instantiated backbone: "
                f"{details}. Use --model-size auto with this checkpoint or choose "
                "a matching model size."
            )

    current = model.backbone.state_dict()
    # Multi-task pretraining checkpoints wrap the shared encoder under a
    # ``backbone.`` module, while downstream checkpoints traditionally store
    # the backbone state dict directly.  Accept both layouts, but only strip
    # the prefix when it produces an exact, shape-compatible target key.
    compatible = {}
    for name, value in state.items():
        candidates = (name[9:], name) if name.startswith("backbone.") else (name,)
        for candidate in candidates:
            if candidate in current and current[candidate].shape == value.shape:
                compatible[candidate] = value
                break
    model.backbone.load_state_dict(compatible, strict=False)
    return len(compatible), len(current) - len(compatible)


def run_epoch(model, loader, device, *, optimizer=None, grad_accumulation=1,
              max_grad_norm=1.0, log_every=50, step_state=None,
              on_optimizer_step=None, max_optimizer_steps=None):
    training = optimizer is not None
    model.train(training)
    if model.freeze_encoder:
        model.backbone.eval()
    total_loss = 0.0
    samples_seen = 0
    labels_all, scores_all = [], []
    optimizer.zero_grad(set_to_none=True) if training else None
    start = time.perf_counter()

    for index, batch in enumerate(loader):
        local_batch_step = index + 1
        if training and step_state is not None:
            step_state["batch_step"] = step_state.get("batch_step", 0) + 1
            batch_step = step_state["batch_step"]
        else:
            batch_step = local_batch_step
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            output = model(input_ids, labels)
            loss = output["loss"]
        if training:
            (loss / grad_accumulation).backward()
            if (index + 1) % grad_accumulation == 0 or index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if step_state is not None:
                    step_state["optimizer_step"] += 1
                if on_optimizer_step is not None:
                    on_optimizer_step(
                        local_batch_step,
                        batch_step,
                        step_state["optimizer_step"] if step_state is not None else None,
                    )
        total_loss += float(loss.detach().float()) * labels.numel()
        samples_seen += labels.numel()
        labels_all.append(labels.detach().cpu().numpy())
        scores_all.append(output["logits"].detach().float().softmax(-1)[:, 1].cpu().numpy())
        if training and local_batch_step % log_every == 0:
            elapsed = max(time.perf_counter() - start, 1e-6)
            print(f"step={local_batch_step}/{len(loader)} loss={total_loss / samples_seen:.5f} "
                  f"samples_s={samples_seen / elapsed:.2f}", flush=True)
        if (training and max_optimizer_steps is not None and step_state is not None
                and step_state.get("optimizer_step", 0) >= max_optimizer_steps):
            break

    labels_np = np.concatenate(labels_all)
    scores_np = np.concatenate(scores_all)
    elapsed_seconds = time.perf_counter() - start
    return {
        "loss": total_loss / len(labels_np),
        "auroc": auroc_binary(labels_np, scores_np),
        "auprc": auprc_binary(labels_np, scores_np),
        "examples": int(len(labels_np)),
        "elapsed_seconds": elapsed_seconds,
    }


class BucketBatchSampler(torch.utils.data.Sampler):
    """Length-grouped batches for dynamic padding.  Training shuffles globally,
    sorts by true length inside megabatches, and reshuffles batch order;
    evaluation deterministically sorts by length.  Each batch is padded to its
    own longest sample by _dynamic_collate instead of to data-max-length."""

    def __init__(self, dataset, batch_size, shuffle, seed, mega_multiplier=50):
        self.lengths = [min(len(seq), dataset.max_length) for seq, _ in dataset.records]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.mega = batch_size * mega_multiplier
        self._epoch = 0

    def __iter__(self):
        n = len(self.lengths)
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self._epoch)
            order = torch.randperm(n, generator=generator).tolist()
            self._epoch += 1
            batches = []
            for start in range(0, n, self.mega):
                chunk = sorted(order[start:start + self.mega],
                               key=lambda i: self.lengths[i])
                for b in range(0, len(chunk), self.batch_size):
                    batches.append(chunk[b:b + self.batch_size])
            random.Random(self.seed * 100003 + self._epoch).shuffle(batches)
        else:
            order = sorted(range(n), key=lambda i: self.lengths[i])
            batches = [order[i:i + self.batch_size]
                       for i in range(0, n, self.batch_size)]
        return iter(batches)

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def _dynamic_collate(batch):
    # Pad id 5 == N, matching _BASE_TO_ID.get(base, 5) in eqtl_data.
    longest = max(item["input_ids"].shape[0] for item in batch)
    input_ids = torch.full((len(batch), longest), 5, dtype=torch.long)
    for index, item in enumerate(batch):
        input_ids[index, :item["input_ids"].shape[0]] = item["input_ids"]
    return {
        "input_ids": input_ids,
        "labels": torch.stack([item["labels"] for item in batch]),
    }


def make_loader(dataset, batch_size, workers, shuffle, dynamic=False, seed=17):
    if not dynamic:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=workers, pin_memory=True,
                          persistent_workers=workers > 0)
    sampler = BucketBatchSampler(dataset, batch_size, shuffle, seed)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=_dynamic_collate,
                      num_workers=workers, pin_memory=True,
                      persistent_workers=workers > 0)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-size", choices=("auto", *available_model_presets()),
                        default="auto",
                        help="architecture preset; auto reads new checkpoint metadata")
    parser.add_argument("--model", default="vericomp",
                        help="model .py path, /path/model.py:ClassName, or built-in alias")
    parser.add_argument("--feature", choices=("backbone", "decoder", "base"), default="backbone")
    parser.add_argument(
        "--dual-strand", action=argparse.BooleanOptionalAction, default=False,
        help="use the reverse-complement path during eQTL backbone inference",
    )
    parser.add_argument("--data-root", default="./data/dnalongbench")
    parser.add_argument("--cell-type", default="Artery_Tibial")
    parser.add_argument("--data-max-length", type=int, default=12_800)
    parser.add_argument("--dynamic-padding", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="bucket by true length; pad each batch to its longest sample instead of data-max-length")
    parser.add_argument("--masked-pooling", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="exclude bottleneck slots whose allocation centre falls on N padding from mean pooling")
    parser.add_argument("--quota-mask-n", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="zero allocation quota on N padding positions")
    parser.add_argument("--loss-cap", type=float, default=None,
                        help="clamp per-sample CE to this value (trimmed loss); clamped samples contribute zero gradient")
    parser.add_argument("--pooling-method", choices=("mean",), default="mean")
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--strict-uncompressed-short",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use unit quota when the input is no longer than max_tokens",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-valid-records", type=int, default=None)
    parser.add_argument("--max-test-records", type=int, default=None)
    parser.add_argument(
        "--max-optimizer-steps", type=int, default=None,
        help="stop after this many optimizer updates for a short throughput probe",
    )
    parser.add_argument("--output-dir",
                        default=str(Path("weights/finetune/eQTL")
                                     / time.strftime("%Y%m%d")))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("eQTL downstream evaluation requires CUDA for the default VeriComp model")
    device = torch.device(args.device)
    if args.batch_size < 1 or args.grad_accumulation < 1:
        raise ValueError("batch size and gradient accumulation must be positive")
    if args.max_optimizer_steps is not None and args.max_optimizer_steps < 1:
        raise ValueError("max_optimizer_steps must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    datasets = {
        subset: EQTLDataset(args.data_root, args.cell_type, subset,
                            args.data_max_length, max_records=limit,
                            dynamic_padding=args.dynamic_padding)
        for subset, limit in (("train", args.max_train_records),
                              ("valid", args.max_valid_records),
                              ("test", args.max_test_records))
    }
    loaders = {
        "train": make_loader(datasets["train"], args.batch_size, args.num_workers, True,
                             dynamic=args.dynamic_padding, seed=args.seed),
        "valid": make_loader(datasets["valid"], args.batch_size, args.num_workers, False,
                             dynamic=args.dynamic_padding, seed=args.seed),
        "test": make_loader(datasets["test"], args.batch_size, args.num_workers, False,
                            dynamic=args.dynamic_padding, seed=args.seed),
    }

    config = model_config_from_checkpoint(
        args.checkpoint,
        args.model_size,
        force_flash_attention=True,
        strict_uncompressed_short=args.strict_uncompressed_short,
        activation_checkpointing=False,
    )
    model = VeriCompEQTLClassifier(config, freeze_encoder=args.freeze_encoder,
                                  model_name=args.model, feature=args.feature,
                                  dual_strand=args.dual_strand,
                                  masked_pooling=args.masked_pooling,
                                  quota_mask_n=args.quota_mask_n,
                                  loss_cap=args.loss_cap).to(device)
    loaded, skipped = load_pretrained_backbone(model, args.checkpoint)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        fused=(device.type == "cuda"),
    )
    steps_per_epoch = max(
        1, (len(loaders["train"]) + args.grad_accumulation - 1) // args.grad_accumulation
    )
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0,
        total_iters=max(1, warmup_steps),
    ) if warmup_steps else None

    output_dir = Path(args.output_dir) / args.cell_type
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = resolve_model_path(args.model)
    write_finetune_config(output_dir, task="eQTL", subtask=args.cell_type,
                          args=args, model_path=model_path, model_config=config,
                          extra={"data_root": str(Path(args.data_root).resolve())})
    print_resolved_config(
        task="eQTL", subtask=args.cell_type, args=args, model_path=model_path,
        model_config=config,
        metadata={
            "checkpoint": {
                "path": str(Path(args.checkpoint).expanduser().resolve()),
                "loaded_backbone_tensors": loaded,
                "skipped_backbone_tensors": skipped,
            },
            "data": {
                "train_examples": len(datasets["train"]),
                "valid_examples": len(datasets["valid"]),
                "test_examples": len(datasets["test"]),
                "max_length": args.data_max_length,
                "bottleneck_pooling": "mean",
                "bottleneck_tokens": min(args.data_max_length, config.max_tokens),
                "num_workers": args.num_workers,
            },
            "optimization": {
                "optimizer": "AdamW",
                "scheduler": "LinearLR warmup",
                "total_steps": total_steps,
                "warmup_ratio": args.warmup_ratio,
                "warmup_steps": warmup_steps,
                "betas": [args.beta1, args.beta2],
                "train_batch_size": args.batch_size,
                "effective_batch_size": args.batch_size * args.grad_accumulation,
            },
            "parameters": {
                "total": sum(p.numel() for p in model.parameters()),
                "trainable": sum(p.numel() for p in trainable),
            },
            "runtime": {
                "device": str(device),
                "torch_version": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "autocast": "bfloat16" if device.type == "cuda" else "disabled",
            },
        },
    )

    best_auroc = -math.inf
    best_valid_metrics = None
    best_test_metrics = None
    best_evaluation = None
    history = []
    step_state = {"batch_step": 0, "optimizer_step": 0, "test_history": []}

    def scheduler_callback(_local_batch_step, _batch_step, optimizer_step):
        if scheduler is not None and optimizer_step <= warmup_steps:
            scheduler.step()

    def evaluate_splits(epoch: int, phase: str):
        nonlocal best_auroc, best_valid_metrics, best_test_metrics, best_evaluation
        model.eval()
        with torch.inference_mode():
            valid_metrics = run_epoch(model, loaders["valid"], device)
            test_metrics = run_epoch(model, loaders["test"], device)
        evaluation = {
            "epoch": epoch,
            "phase": phase,
            "batch_step": step_state["batch_step"],
            "optimizer_step": step_state["optimizer_step"],
            "valid": valid_metrics,
            "test": test_metrics,
        }
        step_state["test_history"].append(evaluation)
        print(json.dumps(evaluation), flush=True)
        if np.isfinite(valid_metrics["auroc"]) and valid_metrics["auroc"] > best_auroc:
            best_auroc = valid_metrics["auroc"]
            best_valid_metrics = valid_metrics
            best_test_metrics = test_metrics
            best_evaluation = evaluation
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "phase": phase, "valid": valid_metrics,
                        "test": test_metrics, "args": vars(args)},
                       output_dir / "best.pt")
        model.train(True)
        if model.freeze_encoder:
            model.backbone.eval()
        return evaluation

    for epoch in range(1, args.epochs + 1):
        midpoint_step = max(1, (len(loaders["train"]) + 1) // 2)
        midpoint_done = False

        def maybe_evaluate_midpoint(local_batch_step, _batch_step, _optimizer_step):
            nonlocal midpoint_done
            if not midpoint_done and local_batch_step >= midpoint_step:
                evaluate_splits(epoch, "midpoint")
                midpoint_done = True

        def on_optimizer_step(local_batch_step, batch_step, optimizer_step):
            scheduler_callback(local_batch_step, batch_step, optimizer_step)
            maybe_evaluate_midpoint(local_batch_step, batch_step, optimizer_step)

        train_metrics = run_epoch(model, loaders["train"], device,
                                  optimizer=optimizer,
                                  grad_accumulation=args.grad_accumulation,
                                  max_grad_norm=args.max_grad_norm,
                                  step_state=step_state,
                                  on_optimizer_step=on_optimizer_step,
                                  max_optimizer_steps=args.max_optimizer_steps)
        if (args.max_optimizer_steps is not None
                and step_state["optimizer_step"] >= args.max_optimizer_steps):
            elapsed = train_metrics["elapsed_seconds"]
            benchmark = {
                "model": args.model,
                "tissue": args.cell_type,
                "context_length_bp": args.data_max_length,
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "optimizer_updates": step_state["optimizer_step"],
                "micro_batch_size": args.batch_size,
                "gradient_accumulation_steps": args.grad_accumulation,
                "effective_batch_size": args.batch_size * args.grad_accumulation,
                "precision": "BF16 autocast; FP32 model parameters and classifier",
                "elapsed_seconds_training_loop": elapsed,
                "seconds_per_optimizer_update": elapsed / step_state["optimizer_step"],
                "samples_per_second": train_metrics["examples"] / elapsed,
                "examples_processed": train_metrics["examples"],
                "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "timing_scope": (
                    "starts before iterating the training DataLoader; includes batch fetching and "
                    "first-use CUDA/kernel overhead, forward/backward, optimizer updates, and the "
                    "runner's small in-memory train-split metric reduction; excludes dataset construction, "
                    "model/checkpoint loading, and validation/test evaluation"
                ),
            }
            benchmark_path = output_dir / "speed_benchmark.json"
            benchmark_path.write_text(json.dumps(benchmark, indent=2) + "\n", encoding="utf-8")
            print(f"[{args.model}] {args.cell_type}: speed_benchmark={json.dumps(benchmark, sort_keys=True)}", flush=True)
            return {"speed_benchmark": benchmark, "speed_benchmark_path": str(benchmark_path)}
        if not midpoint_done:
            evaluate_splits(epoch, "midpoint")
            midpoint_done = True
        end_evaluation = evaluate_splits(epoch, "end")
        record = {"epoch": epoch, "train": train_metrics,
                  "valid": end_evaluation["valid"],
                  "test": end_evaluation["test"],
                  "evaluations": [item for item in step_state["test_history"]
                                  if item["epoch"] == epoch]}
        history.append(record)
        print(json.dumps(record), flush=True)

    best_path = output_dir / "best.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        best_valid_metrics = checkpoint.get("valid", best_valid_metrics)
        best_test_metrics = checkpoint.get("test", best_test_metrics)
    test_metrics = best_test_metrics
    result = {"cell_type": args.cell_type, "valid_best_auroc": best_auroc,
              "best_valid": best_valid_metrics,
              "best_evaluation": best_evaluation,
              "test": test_metrics, "test_history": step_state["test_history"],
              "history": history}
    torch.save({"model": model.state_dict(), "result": result, "args": vars(args)}, output_dir / "last.pt")
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"test": test_metrics, "output_dir": str(output_dir)}), flush=True)


if __name__ == "__main__":
    main()
