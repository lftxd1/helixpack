"""Joint MLM+TISP VeriComp post-training with a shared backbone.

MLM uses fixed 128-Ki-base BED windows and the project's paired-strand span
masking collator. TISP uses the same random training-window distribution,
GeneZip-compatible chromosome-level evaluator, and chr10 validation split as
the downstream TISP runner. Every optimizer update includes one batch from
each task and minimizes their explicitly weighted sum.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from finetune.TISP.external_tisp import evaluate_tisp_chromosome
from finetune.TISP.tisp import VeriCompTISPModel
from pretrain.run_config import print_resolved_config, write_run_config
from pretrain.train import (
    AsyncLogWriter,
    build_optimizer,
    make_loader,
    masked_accuracy,
    setup_distributed,
)
from vericomp import VeriCompConfig
from vericomp.data import BEDGenomeDataset
from vericomp.tisp_data import TISPRandomDataset, TISPWindowDataset


DEFAULT_CHECKPOINT = (
    "/gpfs_data/Projects/HelixPack/weights/pretrain/20260814/"
    "vericomp-residual-skip-30k/checkpoint-010000.pt"
)
DEFAULT_MLM_FASTA = "/gpfs_data/Projects/data/hg38/hg38.ml.fa"
DEFAULT_MLM_BED = "/gpfs_data/Projects/data/hg38/human-sequences.bed"
DEFAULT_TISP_ROOT = "/gpfs_data/Projects/HelixPack/data/dnalongbench"


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--mlm-fasta", default=DEFAULT_MLM_FASTA)
    parser.add_argument("--mlm-bed", default=DEFAULT_MLM_BED)
    parser.add_argument("--tisp-root", default=DEFAULT_TISP_ROOT)
    parser.add_argument(
        "--output-dir",
        default=(
        "weights/pretrain/20260920/"
            "vericomp-mlm-tisp-128k-joint-posttrain-two-gpu-lr2e-4-to1e-4-tisp-ratio0p2-b6-30k"
        ),
    )
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--mlm-context-length", type=int, default=131_072)
    parser.add_argument("--mlm-batch-size", type=int, default=6,
                        help="per-GPU MLM batch for the 128 Ki context")
    parser.add_argument("--tisp-batch-size", type=int, default=6,
                        help="per-GPU TISP batch used alongside the 128 Ki MLM batch")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--mlm-workers", type=int, default=2)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="initial learning rate for the joint MLM+TISP schedule")
    parser.add_argument("--lr-end", type=float, default=1e-4,
                        help="final learning rate for linear decay")
    parser.add_argument("--lr-schedule", choices=("linear", "constant"), default="linear",
                        help="linear decay from --lr to --lr-end, or constant LR")
    parser.add_argument("--mlm-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--tisp-loss-ratio", "--tisp-loss-weight", dest="tisp_loss_ratio",
        type=float, default=0.2,
        help="target ratio of dynamically weighted TISP KL to MLM cross-entropy",
    )
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="optional warmup to --lr before linear decay; default starts at --lr")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--tisp-router-loss-coef", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--smoke-test", action="store_true",
                        help="run one joint MLM+TISP backward pass, then exit")
    return parser.parse_args()


def _seed(seed: int, rank: int) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _load_config_and_state(path: str, activation_checkpointing: bool):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source_config = checkpoint.get("model_config")
    if isinstance(source_config, dict):
        valid = {field.name for field in fields(VeriCompConfig)}
        values = {name: value for name, value in source_config.items() if name in valid}
        values.update(
            force_flash_attention=True,
            activation_checkpointing=bool(activation_checkpointing),
        )
        config = VeriCompConfig(**values)
    else:
        # This 2026-08 checkpoint predates embedded model_config metadata.
        # Its checkpoint args contain the remaining non-default architecture
        # switches, while d_model and module shapes are verified on load.
        source_args = checkpoint.get("args", {})
        values = {
            name: source_args[name]
            for name in (
                "num_layers", "allocation_mode", "strict_uncompressed_short",
                "duplex_mode",
            )
            if name in source_args
        }
        values.update(
            force_flash_attention=True,
            activation_checkpointing=bool(activation_checkpointing),
        )
        config = VeriCompConfig(**values)
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no model state dict: {path}")
    return config, checkpoint, state


class MultiTaskVeriCompTISPModel(VeriCompTISPModel):
    """Shared backbone with dynamically loss-balanced MLM and TISP objectives."""

    def __init__(self, *args, mlm_loss_weight: float, tisp_loss_ratio: float,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.mlm_loss_weight = float(mlm_loss_weight)
        self.tisp_loss_ratio = float(tisp_loss_ratio)

    @staticmethod
    def _global_mean_for_weight(loss: torch.Tensor) -> torch.Tensor:
        """Detached global-batch mean, identical on every DDP rank."""
        value = loss.detach().to(dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value /= dist.get_world_size()
        return value.to(dtype=loss.dtype)

    def forward(self, mlm_batch, tisp_batch):
        # Both task losses are constructed in the same DDP forward so both
        # heads and all shared-backbone gradients are reduced together.
        mlm_output = self.backbone(
            input_ids_pos=mlm_batch["input_ids_pos"],
            input_ids_rc=mlm_batch["input_ids_rc"],
            labels=mlm_batch["labels"],
        )
        tisp_output = super().forward(
            tisp_batch["input_ids"], tisp_batch.get("target")
        )
        mlm_loss = mlm_output["loss_mlm"].float()
        tisp_loss = tisp_output["loss_task"].float()
        # Each task keeps its native router penalty. MLM uses the coefficient
        # stored in the pretrained config; TISP uses its 1e-4 fine-tune value.
        mlm_objective = mlm_output["loss"].float()
        tisp_objective = tisp_output["loss"].float()
        weighted_mlm = self.mlm_loss_weight * mlm_objective
        # Balance the primary task losses (MLM cross-entropy and TISP KL), not
        # their auxiliary router penalties. The detached, globally averaged
        # coefficient is shared by all DDP ranks and yields the requested
        # scalar-loss ratio exactly for every optimizer batch.
        global_mlm_loss = self._global_mean_for_weight(mlm_loss)
        global_tisp_loss = self._global_mean_for_weight(tisp_loss)
        if (not bool(torch.isfinite(global_mlm_loss))
                or not bool(torch.isfinite(global_tisp_loss))
                or float(global_mlm_loss) <= 0.0
                or float(global_tisp_loss) <= 0.0):
            raise RuntimeError(
                "cannot dynamically balance non-finite or non-positive global TISP loss"
            )
        tisp_loss_weight = (
            self.tisp_loss_ratio * self.mlm_loss_weight
            * global_mlm_loss / global_tisp_loss
        ).detach()
        weighted_tisp_task = tisp_loss_weight * tisp_loss
        tisp_router_term = tisp_output["loss_router_term"].float()
        weighted_tisp = weighted_tisp_task + tisp_router_term
        return {
            "loss": weighted_mlm + weighted_tisp,
            "mlm": mlm_output,
            "tisp": tisp_output,
            "mlm_loss": mlm_loss,
            "tisp_loss": tisp_loss,
            "mlm_objective": mlm_objective,
            "tisp_objective": tisp_objective,
            "mlm_router_loss": mlm_output["loss_router"].float(),
            "tisp_router_loss": tisp_output["router"].float(),
            "tisp_router_term": tisp_router_term,
            "weighted_mlm_loss": weighted_mlm,
            "weighted_mlm_task_loss": self.mlm_loss_weight * mlm_loss,
            "weighted_tisp_loss": weighted_tisp,
            "weighted_tisp_task_loss": weighted_tisp_task,
            "dynamic_tisp_loss_weight": tisp_loss_weight,
            "tisp_to_mlm_loss_ratio": (
                tisp_loss_weight * global_tisp_loss
                / (self.mlm_loss_weight * global_mlm_loss)
            ),
        }


class _TISPValidationView(nn.Module):
    """Adapt the multi-task dispatch API to the chromosome evaluator."""

    def __init__(self, model: MultiTaskVeriCompTISPModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids, target=None):
        # Bypass the joint-task dispatch and call the inherited TISP path
        # directly for chromosome-level validation/inference.
        return VeriCompTISPModel.forward(self.model, input_ids, target)


def _atomic_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _save_model(model, optimizer, config, args, output_dir, step, best_valid,
                valid_metrics, history, *, best: bool):
    payload = {
        "task": "MLM+TISP",
        "model": model.state_dict(),
        "model_config": asdict(model.backbone.config),
        "step": int(step),
        "training_method": "joint MLM+TISP weighted-sum update; both tasks every step",
        "mlm_loss_weight": float(args.mlm_loss_weight),
        "target_tisp_to_mlm_loss_ratio": float(args.tisp_loss_ratio),
        "mlm_updates": int(step),
        "tisp_updates": int(step),
        "best_tisp_valid_avg_pcc": float(best_valid),
        "valid": valid_metrics,
        "history": history,
        "arguments": vars(args),
    }
    if best:
        _atomic_save(payload, output_dir / "best.pt")
    _atomic_save(payload, output_dir / "last.pt")
    result = {
        "task": "MLM+TISP",
        "step": int(step),
        "best_tisp_valid_avg_pcc": float(best_valid),
        "valid": valid_metrics,
        "history": history,
    }
    _atomic_json(result, output_dir / "metrics.json")


def _reduce_values(values, device, *, average: bool):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if average:
            tensor /= dist.get_world_size()
    return tensor.cpu().tolist()


def _current_lr(optimizer):
    return float(optimizer.param_groups[0]["lr"])


def _learning_rate_for_step(args, step: int) -> float:
    """Return this optimizer step's LR, with optional warmup then linear decay."""
    if args.lr_schedule == "constant":
        if args.warmup_steps:
            return args.lr * min(1.0, step / args.warmup_steps)
        return args.lr

    if args.warmup_steps and step <= args.warmup_steps:
        return args.lr * step / args.warmup_steps
    decay_start = max(1, args.warmup_steps)
    decay_steps = max(1, args.steps - decay_start)
    progress = min(1.0, max(0.0, (step - decay_start) / decay_steps))
    return args.lr + (args.lr_end - args.lr) * progress


def main():
    args = parse_args()
    distributed, rank, world_size, device = setup_distributed()
    writer = AsyncLogWriter(rank == 0)
    try:
        if not torch.cuda.is_available() or device.type != "cuda":
            raise RuntimeError("multi-task post-training requires CUDA")
        if world_size != 2:
            raise RuntimeError(f"this run is configured for exactly two GPUs; got {world_size}")
        if args.mlm_context_length != 131_072:
            raise ValueError("this recipe fixes the MLM context at 131,072 bases (128 Ki)")
        if (args.steps < 1 or args.eval_every < 1 or args.warmup_steps < 0
                or args.lr <= 0 or args.lr_end <= 0):
            raise ValueError("steps/eval interval and learning rates must be positive; warmup non-negative")
        if args.warmup_steps > args.steps:
            raise ValueError("warmup cannot exceed total steps")
        if args.mlm_loss_weight <= 0 or args.tisp_loss_ratio <= 0:
            raise ValueError("MLM loss weight and target TISP/MLM ratio must be positive")

        _seed(args.seed, rank)
        output_dir = Path(args.output_dir).resolve()
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        if distributed:
            dist.barrier()

        model_config, source_checkpoint, source_state = _load_config_and_state(
            args.checkpoint, args.activation_checkpointing
        )
        model = MultiTaskVeriCompTISPModel(
            model_config,
            freeze_encoder=False,
            model_name="vericomp",
            feature="decoder",
            router_loss_coef=args.tisp_router_loss_coef,
            mlm_loss_weight=args.mlm_loss_weight,
            tisp_loss_ratio=args.tisp_loss_ratio,
            dual_strand=True,
            all_bf16=False,
        ).to(device)
        current_state = model.backbone.state_dict()
        compatible = {
            name: value for name, value in source_state.items()
            if name in current_state and current_state[name].shape == value.shape
        }
        missing = sorted(set(current_state) - set(compatible))
        unexpected = sorted(set(source_state) - set(compatible))
        if missing or unexpected:
            raise RuntimeError(
                "pretrained checkpoint/backbone mismatch: "
                f"loaded {len(compatible)}/{len(current_state)} backbone tensors, "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        model.backbone.load_state_dict(compatible, strict=True)
        loaded_tensors = len(compatible)
        del source_checkpoint, source_state, compatible
        model.use_bf16 = True

        ddp_model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            gradient_as_bucket_view=True,
            find_unused_parameters=False,
        )
        optimizer = build_optimizer(
            ddp_model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay, fused=True,
        )

        mlm_dataset = BEDGenomeDataset(
            args.mlm_bed,
            args.mlm_fasta,
            args.mlm_context_length,
            sequence_lengths=[args.mlm_context_length],
            split="train",
            seed=args.seed,
            random_window_sampling=True,
        )
        mlm_loader, mlm_sampler = make_loader(
            mlm_dataset,
            world_size=world_size,
            rank=rank,
            batch_sizes={args.mlm_context_length: args.mlm_batch_size},
            seed=args.seed,
            shuffle=True,
            drop_last=True,
            workers=args.mlm_workers,
            equal_length_sampling=True,
        )
        mlm_sampler.set_epoch(0)
        mlm_iterator = iter(mlm_loader)

        # Same training split/window sampler as finetune/TISP/vericomp_dnalongbench.py.
        tisp_dataset = TISPRandomDataset(
            args.tisp_root, 0, seed=3 + rank * 1_000_003
        )
        tisp_loader = DataLoader(
            tisp_dataset,
            batch_size=args.tisp_batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
        tisp_iterator = iter(tisp_loader)

        param_count = sum(parameter.numel() for parameter in model.parameters())
        run_config = {
            "start_time_utc": datetime.now(timezone.utc).isoformat(),
            "task": "MLM+TISP",
            "objective": "shared-backbone post-training; downstream TISP inference from validation-best checkpoint",
            "source_checkpoint": str(Path(args.checkpoint).resolve()),
            "loaded_backbone_tensors": loaded_tensors,
            "total_parameters": param_count,
            "world_size": world_size,
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "training_method": "joint multi-task training; every optimizer step computes both task losses and updates one shared backbone",
            "loss_objective": (
                "mlm_weight * (MLM cross-entropy + MLM router term) + "
                "dynamic_weight * TISP pseudo-Poisson KL + TISP router term"
            ),
            "mlm_loss_weight": args.mlm_loss_weight,
            "target_tisp_to_mlm_loss_ratio": args.tisp_loss_ratio,
            "tisp_weighting": "per-step dynamic global-batch ratio; router auxiliary terms excluded",
            "task_weight_rationale": (
                "At every optimizer step, choose a detached coefficient so the "
                "global-mean weighted TISP pseudo-Poisson KL equals the target "
                "ratio times the global-mean weighted MLM cross-entropy. Router "
                "auxiliary terms remain separate. This balances scalar losses, "
                "not gradient norms."
            ),
            "steps": args.steps,
            "mlm_optimizer_updates": args.steps,
            "tisp_optimizer_updates": args.steps,
            "mlm_context_length": args.mlm_context_length,
            "mlm_fasta": str(Path(args.mlm_fasta).resolve()),
            "mlm_bed": str(Path(args.mlm_bed).resolve()),
            "mlm_split": "train",
            "mlm_masking": "SpanMLMCollator, mask_prob=0.15, mean_span_length=8, paired reverse-complement masks",
            "mlm_batch_per_gpu": args.mlm_batch_size,
            "mlm_global_batch": args.mlm_batch_size * world_size,
            "tisp_root": str(Path(args.tisp_root).resolve()),
            "tisp_train_set": "TISPRandomDataset, random 100-kb windows from chromosomes excluding 8, 9, 10; N fraction <= 0.30",
            "tisp_sequence_length": 100_000,
            "tisp_batch_per_gpu": args.tisp_batch_size,
            "tisp_global_batch": args.tisp_batch_size * world_size,
            "tisp_training_precision": "BF16 autocast; TISP loss FP32",
            "mlm_router_aux_loss_coef": float(model_config.router_aux_loss_coef),
            "tisp_validation": "full chr10, GeneZip-compatible chromosome stitching/deduplication, five strand-paired PCCs",
            "validation_every_optimizer_steps": args.eval_every,
            "validation_best": "chr10 average PCC; test chromosomes not used for selection",
            "evaluation_batch_size": args.eval_batch_size,
            "learning_rate": args.lr,
            "learning_rate_end": args.lr_end,
            "learning_rate_schedule": args.lr_schedule,
            "warmup_steps": args.warmup_steps,
            "scheduler": (
                "optional warmup to initial LR, then linear decay to final LR"
                if args.lr_schedule == "linear" else
                "optional linear warmup, then constant LR"
            ),
            "weight_decay": args.weight_decay,
            "gradient_clip": args.max_grad_norm,
            "tisp_router_loss_coef": args.tisp_router_loss_coef,
            "activation_checkpointing": args.activation_checkpointing,
            "seed": args.seed,
            "arguments": vars(args),
        }
        if rank == 0:
            print_resolved_config(
                entrypoint="pretrain.multitask_mlm_tisp",
                args=args,
                model_config=model_config,
                runtime={
                    "world_size": world_size,
                    "device": torch.cuda.get_device_name(device),
                    "loaded_backbone_tensors": loaded_tensors,
                    "total_parameters": param_count,
                },
                extra={"resolved_run_config": run_config},
            )
            write_run_config(
                output_dir,
                entrypoint="pretrain.multitask_mlm_tisp",
                model_path=Path(__file__).resolve().parents[1] / "vericomp/modeling.py",
                model_class="MultiTaskVeriCompTISPModel",
                model_config=model_config,
                args=args,
                runtime=run_config,
                extra={"data": {"mlm_fasta": args.mlm_fasta,
                                "mlm_bed": args.mlm_bed,
                                "tisp_root": args.tisp_root}},
            )
        if distributed:
            dist.barrier()

        if args.smoke_test:
            optimizer.zero_grad(set_to_none=True)
            mlm_batch = next(mlm_iterator)
            mlm_batch = {key: value.to(device, non_blocking=True)
                         for key, value in mlm_batch.items()}
            tisp_batch = next(tisp_iterator)
            tisp_batch = {key: value.to(device, non_blocking=True)
                          for key, value in tisp_batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = ddp_model(mlm_batch, tisp_batch)
            if not torch.isfinite(output["loss"]):
                raise RuntimeError("non-finite joint loss in smoke test")
            output["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                ddp_model.parameters(), args.max_grad_norm
            )
            optimizer.step()
            if rank == 0:
                print(json.dumps({
                    "smoke_test": "joint_mlm_tisp",
                    "loss": float(output["loss"].detach()),
                    "mlm_loss": float(output["mlm_loss"].detach()),
                    "tisp_loss": float(output["tisp_loss"].detach()),
                    "weighted_mlm_loss": float(output["weighted_mlm_loss"].detach()),
                    "weighted_mlm_task_loss": float(output["weighted_mlm_task_loss"].detach()),
                    "weighted_tisp_loss": float(output["weighted_tisp_loss"].detach()),
                    "weighted_tisp_task_loss": float(output["weighted_tisp_task_loss"].detach()),
                    "dynamic_tisp_loss_weight": float(output["dynamic_tisp_loss_weight"].detach()),
                    "tisp_to_mlm_loss_ratio": float(output["tisp_to_mlm_loss_ratio"].detach()),
                    "mlm_objective": float(output["mlm_objective"].detach()),
                    "tisp_objective": float(output["tisp_objective"].detach()),
                    "mlm_router_loss": float(output["mlm_router_loss"].detach()),
                    "tisp_router_loss": float(output["tisp_router_loss"].detach()),
                    "tisp_router_term": float(output["tisp_router_term"].detach()),
                    "grad_norm": float(grad_norm.detach()),
                    "mlm_context_length": args.mlm_context_length,
                    "tisp_context_length": 100_000,
                    "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
                }, ensure_ascii=False), flush=True)
            model.eval()
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                validation_output = _TISPValidationView(model)(
                    tisp_batch["input_ids"], tisp_batch["target"]
                )
            if not torch.isfinite(validation_output["loss"]):
                raise RuntimeError("non-finite TISP validation-view loss in smoke test")
            if rank == 0:
                print(json.dumps({
                    "smoke_test": "tisp_validation_view",
                    "loss": float(validation_output["loss"].detach()),
                    "prediction_shape": list(validation_output["pred"].shape),
                }, ensure_ascii=False), flush=True)
            if distributed:
                dist.barrier()
            return

        valid_dataset = (
            TISPWindowDataset(args.tisp_root, ["10"]) if rank == 0 else None
        )
        valid_loader = (
            DataLoader(valid_dataset, batch_size=args.eval_batch_size,
                       shuffle=False, num_workers=args.eval_workers,
                       pin_memory=True, persistent_workers=args.eval_workers > 0)
            if rank == 0 else None
        )
        test_dataset = (
            TISPWindowDataset(args.tisp_root, ["8", "9"]) if rank == 0 else None
        )
        test_loader = (
            DataLoader(test_dataset, batch_size=args.eval_batch_size,
                       shuffle=False, num_workers=args.eval_workers,
                       pin_memory=True, persistent_workers=args.eval_workers > 0)
            if rank == 0 else None
        )

        best_valid = -math.inf
        history = []
        mlm_epoch = 0
        start_time = time.perf_counter()
        for step in range(1, args.steps + 1):
            lr = _learning_rate_for_step(args, step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            try:
                mlm_batch = next(mlm_iterator)
            except StopIteration:
                mlm_epoch += 1
                mlm_sampler.set_epoch(mlm_epoch)
                mlm_iterator = iter(mlm_loader)
                mlm_batch = next(mlm_iterator)
            mlm_batch = {key: value.to(device, non_blocking=True)
                         for key, value in mlm_batch.items()}
            try:
                tisp_batch = next(tisp_iterator)
            except StopIteration:
                tisp_iterator = iter(tisp_loader)
                tisp_batch = next(tisp_iterator)
            tisp_batch = {key: value.to(device, non_blocking=True)
                          for key, value in tisp_batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = ddp_model(mlm_batch, tisp_batch)
                loss = output["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite joint loss at step={step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                ddp_model.parameters(), args.max_grad_norm
            )
            optimizer.step()

            if step == 1 or step % 50 == 0:
                mlm_output = output["mlm"]
                local_correct, local_masked = masked_accuracy(
                    mlm_output["logits"].detach(), mlm_batch["labels"]
                )
                reduced = _reduce_values([
                    float(loss.detach()), float(output["mlm_loss"].detach()),
                    float(output["tisp_loss"].detach()),
                    float(output["weighted_mlm_loss"].detach()),
                    float(output["weighted_tisp_loss"].detach()),
                    float(output["mlm_objective"].detach()),
                    float(output["tisp_objective"].detach()),
                    float(output["mlm_router_loss"].detach()),
                    float(output["tisp_router_loss"].detach()),
                    float(output["tisp_router_term"].detach()),
                    float(output["weighted_tisp_task_loss"].detach()),
                    float(output["dynamic_tisp_loss_weight"].detach()),
                    float(output["tisp_to_mlm_loss_ratio"].detach()),
                    float(output["weighted_mlm_task_loss"].detach()),
                ], device, average=True)
                correct, masked = _reduce_values(
                    [local_correct, local_masked], device, average=False
                )
                record = {
                    "step": step, "tasks": ["MLM", "TISP"],
                    "train_loss": reduced[0], "mlm_loss": reduced[1],
                    "tisp_loss": reduced[2], "weighted_mlm_loss": reduced[3],
                    "weighted_tisp_loss": reduced[4], "mlm_objective": reduced[5],
                    "tisp_objective": reduced[6], "mlm_router_loss": reduced[7],
                    "tisp_router_loss": reduced[8], "tisp_router_term": reduced[9],
                    "weighted_tisp_task_loss": reduced[10],
                    "dynamic_tisp_loss_weight": reduced[11],
                    "tisp_to_mlm_loss_ratio": reduced[12],
                    "weighted_mlm_task_loss": reduced[13],
                    "mlm_accuracy": correct / max(1.0, masked),
                    "masked_tokens_global": masked,
                    "mlm_global_examples": args.mlm_batch_size * world_size,
                    "tisp_global_examples": args.tisp_batch_size * world_size,
                    "lr": lr, "grad_norm": float(grad_norm.detach()),
                    "elapsed_s": time.perf_counter() - start_time,
                }
                if rank == 0:
                    writer.log(json.dumps(record, ensure_ascii=False))

            if step % args.eval_every == 0 or step == args.steps:
                if distributed:
                    dist.barrier()
                if rank == 0:
                    view = _TISPValidationView(model)
                    metrics = evaluate_tisp_chromosome(view, valid_loader, device)
                    record = {"step": step, "valid": metrics}
                    history.append(record)
                    is_best = (
                        math.isfinite(metrics["avg_pcc"])
                        and metrics["avg_pcc"] > best_valid
                    )
                    if is_best:
                        best_valid = float(metrics["avg_pcc"])
                    writer.log(json.dumps(record, ensure_ascii=False))
                    _save_model(
                        model, optimizer, model_config, args, output_dir, step,
                        best_valid, metrics, history, best=is_best,
                    )
                if distributed:
                    score = torch.tensor(
                        [best_valid if rank == 0 else 0.0],
                        dtype=torch.float64, device=device,
                    )
                    dist.broadcast(score, src=0)
                    best_valid = float(score.item())
                    dist.barrier()

        if rank == 0 and test_loader is not None:
            best_path = output_dir / "best.pt"
            best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
            model.load_state_dict(best_payload["model"], strict=True)
            test_metrics = evaluate_tisp_chromosome(
                _TISPValidationView(model), test_loader, device
            )
            result = {
                "task": "MLM+TISP",
                "model": "VeriComp",
                "best_step": int(best_payload["step"]),
                "best_valid_avg_pcc": float(best_payload["best_tisp_valid_avg_pcc"]),
                "test_validation_best": test_metrics,
                "history": history,
            }
            _atomic_json(result, output_dir / "metrics.json")
            writer.log(json.dumps({"test_validation_best": test_metrics,
                                   "best_step": best_payload["step"]}, ensure_ascii=False))
        if distributed:
            dist.barrier()
    finally:
        writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
