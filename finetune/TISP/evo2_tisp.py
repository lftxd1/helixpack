"""Frozen Evo2 backbone with a trainable FP32 TISP readout.

The backbone runs in its native BF16 under torch.no_grad; only the LayerNorm
and Linear readout receive gradients and optimizer updates.  Multiple ranks can
share a run through torch.distributed, in which case the tiny readout gradients
are all-reduced once per optimizer step.
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
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

from finetune.TISP.chromosome_eval import ChromosomeTISP, evaluate_chromosome_level
from vericomp.tisp_data import TISP_SEQUENCE_LENGTH, TISPRandomDataset

TRACKS = 10
ID_MAP = torch.tensor([65, 67, 71, 84, 0, 78], dtype=torch.long)  # A C G T . N
EPS = 1e-5


def pseudo_poisson_kl(target: torch.Tensor, rates: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(target) & torch.isfinite(rates)
    if not bool(valid.any()):
        return rates.sum() * 0.0
    t = target[valid]
    r = rates[valid]
    return (t * torch.log((t + EPS) / (r + EPS)) + r - t).mean()


class Readout(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, TRACKS)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(features))


class Evo2TISPModel(nn.Module):
    def __init__(self, checkpoint: str, width: int):
        super().__init__()
        from evo2.models import Evo2

        self.wrapper = Evo2("evo2_1b_base", local_path=checkpoint)
        self.backbone = self.wrapper.model
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        self.captured: dict[str, torch.Tensor] = {}
        self.backbone.norm.register_forward_hook(
            lambda _m, _i, out: self.captured.__setitem__("norm", out.detach())
        )
        self.readout = Readout(width)
        self.use_bf16 = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def features(self, input_ids: torch.Tensor) -> torch.Tensor:
        ascii_ids = ID_MAP.to(input_ids.device)[input_ids]
        with torch.no_grad():
            self.wrapper(ascii_ids)
        return self.captured.pop("norm").to(torch.float32)

    def forward(self, input_ids: torch.Tensor, target: torch.Tensor | None = None):
        log_rates = self.readout(self.features(input_ids))
        result = {"log_rates": log_rates, "pred": torch.exp(log_rates)}
        if target is not None:
            result["loss"] = pseudo_poisson_kl(target.float(), result["pred"])
        return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--train-steps", type=int, default=250)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--max-valid-samples", type=int, default=600)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--scheduler-start-factor", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--init-head-bias", choices=("none", "mean"), default="mean")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def init_distributed() -> tuple[int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return 0, 1
    dist.init_process_group("nccl")
    return int(os.environ["RANK"]), world


def main() -> None:
    args = parse_args()
    rank, world = init_distributed()
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = Evo2TISPModel(args.checkpoint, args.width).to(device)
    readout = model.readout
    if world > 1:
        readout = nn.parallel.DistributedDataParallel(
            readout, device_ids=[device.index or 0], broadcast_buffers=False
        )
    model.readout = readout  # evaluation calls the same wrapper

    optimizer = torch.optim.AdamW(readout.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_steps = max(1, int(args.train_steps * args.warmup_ratio))

    def lr_scale(step: int) -> float:
        if step < warmup_steps:
            return args.scheduler_start_factor + (1.0 - args.scheduler_start_factor) * (step + 1) / warmup_steps
        decay_steps = max(1, args.train_steps - warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / decay_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)

    def forward_with_readout(input_ids, target):
        features = model.features(input_ids)
        log_rates = readout(features)
        pred = torch.exp(log_rates)
        return {"log_rates": log_rates, "pred": pred, "loss": pseudo_poisson_kl(target.float(), pred)}

    train_ds = TISPRandomDataset(args.data_root, 0, seed=args.seed + 1000 * rank)
    valid_ds = ChromosomeTISP(args.data_root, ["chr10", "10"])
    if args.max_valid_samples:
        valid_ds.windows = valid_ds.windows[: args.max_valid_samples]
    test_ds = ChromosomeTISP(args.data_root, ["chr8", "8", "chr9", "9"])
    if args.max_test_samples:
        test_ds.windows = test_ds.windows[: args.max_test_samples]

    train_loader = DataLoader(
        train_ds, batch_size=1, num_workers=args.num_workers,
        pin_memory=args.num_workers > 0, prefetch_factor=4 if args.num_workers else None,
        persistent_workers=args.num_workers > 0)
    valid_loader = DataLoader(valid_ds, batch_size=args.eval_batch_size, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, num_workers=2, pin_memory=True)

    readout_state = readout.module if world > 1 else readout
    if args.init_head_bias == "mean":
        totals = None
        samples = 0
        for index, batch in enumerate(train_loader):
            if index >= 4:
                break
            target = batch["target"].to(device).float()
            batch_sum = target.sum(dim=(0, 1))
            totals = batch_sum if totals is None else totals + batch_sum
            samples += target.shape[0] * target.shape[1]
        totals = totals / max(samples, 1)
        if world > 1:
            dist.all_reduce(totals)
            totals = totals / world
        per_track_mean = totals.clamp_min(1e-8)
        with torch.no_grad():
            readout_state.head.bias.copy_(torch.log(per_track_mean))
        if rank == 0:
            print(json.dumps({"event": "head_bias_init",
                              "per_track_mean": [float(v) for v in per_track_mean.cpu()]}), flush=True)

    if rank == 0:
        config = {
            "task": "TISP",
            "model": "Evo2-1B-base",
            "checkpoint": args.checkpoint,
            "backbone": "frozen; native BF16, forward under no_grad",
            "readout": "trainable FP32 LayerNorm + Linear(1920, 10)",
            "loss": "pseudo_poisson_kl in FP32",
            "sequence_length": TISP_SEQUENCE_LENGTH,
            "train_steps": args.train_steps,
            "gradient_accumulation": args.grad_accumulation,
            "world_size": world,
            "effective_batch_size": args.grad_accumulation * world,
            "learning_rate": args.lr,
            "warmup_ratio": args.warmup_ratio,
            "scheduler": "linear",
            "seed": args.seed,
            "valid_windows": len(valid_ds.windows),
            "test_windows": len(test_ds.windows),
            "start_time_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        }
        (out_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    best = -math.inf
    step = 0
    micro = 0
    running = 0.0
    seen = 0
    started = time.time()
    while step < args.train_steps:
        for batch in train_loader:
            model.train()
            target = batch["target"].to(device, non_blocking=True)
            accumulating = micro + 1 < args.grad_accumulation
            context = readout.no_sync() if (world > 1 and accumulating) else torch.enable_grad()
            with context:
                output = forward_with_readout(batch["input_ids"].to(device, non_blocking=True), target)
                loss = output["loss"] / args.grad_accumulation
                loss.backward()
            running += float(loss.detach()) * args.grad_accumulation
            seen += 1
            micro += 1
            if micro < args.grad_accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(readout.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            micro = 0
            step += 1
            if rank == 0 and step % args.log_every == 0:
                print(json.dumps({
                    "step": step,
                    "train_loss": running / (args.log_every * args.grad_accumulation),
                    "lr": optimizer.param_groups[0]["lr"],
                    "windows": seen,
                    "elapsed_min": round((time.time() - started) / 60, 1),
                }), flush=True)
                running = 0.0
            if step % args.eval_every == 0 or step == args.train_steps:
                if rank == 0:
                    metrics = evaluate_chromosome_level(model, valid_loader, device, valid_ds)
                    print(json.dumps({"step": step, "valid": metrics}), flush=True)
                    if metrics["avg_pcc"] > best:
                        best = metrics["avg_pcc"]
                        state = readout.module if world > 1 else readout
                        torch.save({"readout": state.state_dict(), "step": step, "valid": metrics},
                                   out_dir / "best.pt")
                if world > 1:
                    dist.barrier()
            if step >= args.train_steps:
                break

    if rank == 0:
        state = readout.module if world > 1 else readout
        torch.save({"readout": state.state_dict(), "step": step}, out_dir / "last.pt")
    if world > 1:
        dist.barrier()
    result = {"valid_best_avg_pcc": best, "train_windows": seen, "steps": step,
              "world_size": world, "elapsed_min": round((time.time() - started) / 60, 1)}
    if rank == 0 and not args.skip_test and (out_dir / "best.pt").exists():
        checkpoint = torch.load(out_dir / "best.pt", map_location=device)
        state = readout.module if world > 1 else readout
        state.load_state_dict(checkpoint["readout"])
        result["test_validation_best"] = evaluate_chromosome_level(model, test_loader, device, test_ds)
        print(json.dumps({"test": result["test_validation_best"]}), flush=True)
    if rank == 0:
        (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({"output_dir": str(out_dir), "valid_best_avg_pcc": best}), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
