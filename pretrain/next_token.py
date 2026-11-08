"""Causal Next-Token pre-training for VeriComp.

This entry point warm-starts the model weights from an existing VeriComp
checkpoint, but deliberately resets the optimizer and scheduler.  Unlike the
MLM trainer, the collator does no masking: it shifts the positive strand one
position to form the next-token targets and shifts the reverse-complement
strand in the paired direction.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from pretrain.train import (
    AsyncLogWriter,
    assert_same_length,
    build_optimizer,
    build_scheduler,
    format_duration,
    parse_batch_sizes,
    save_checkpoint,
    setup_distributed,
)
from vericomp.model_presets import available_model_presets, model_config_from_checkpoint
from vericomp.causal_modeling import CausalVeriCompNextTokenModel
from vericomp.data import BEDGenomeDataset, DistributedLengthBatchSampler
from pretrain.run_config import print_resolved_config, write_run_config


DEFAULT_LENGTHS = "1024,2048,4096,8192,16384,32768,65536,131072"
DEFAULT_BATCH_SIZES = "1024:128,2048:64,4096:32,8192:32,16384:16,32768:16,65536:8,131072:8"


class NextTokenCollator:
    """Create paired causal inputs without MLM masking or padding."""

    def __call__(self, batch):
        pos = torch.stack([item["input_ids_pos"] for item in batch])
        rc = torch.stack([item["input_ids_rc"] for item in batch])
        model_input = pos[:, :-1]
        labels = pos[:, 1:].clone()
        # This repository uses 0..3 for A/C/G/T and 5 for unknown bases.
        labels[(labels < 0) | (labels > 3)] = -100
        # rc[:, 1:] is exactly reverse-complement(model_input).
        return {
            "input_ids_pos": model_input,
            "input_ids_rc": rc[:, 1:],
            "labels": labels,
        }


def make_loader(dataset, *, world_size, rank, batch_sizes, seed, shuffle,
                drop_last, workers):
    sampler = DistributedLengthBatchSampler(
        dataset,
        world_size=world_size,
        rank=rank,
        per_rank_batch_size=batch_sizes,
        drop_last=drop_last,
        shuffle=shuffle,
        seed=seed,
        equal_length_sampling=True,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=NextTokenCollator(),
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    return loader, sampler


def reduce_sum(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def batch_metrics(output, labels, device):
    valid = labels.ne(-100)
    token_count = int(valid.sum().item())
    correct = int((output["logits"].argmax(dim=-1).eq(labels) & valid).sum().item())
    loss = float(output["loss_mlm"].detach().float().item())
    router = float(output["loss_router"].detach().float().item())
    total = float(output["loss"].detach().float().item())
    return {
        "loss_sum": reduce_sum(loss * max(1, token_count), device),
        "total_sum": reduce_sum(total * max(1, token_count), device),
        "router_sum": reduce_sum(router * max(1, token_count), device),
        "tokens": reduce_sum(token_count, device),
        "correct": reduce_sum(correct, device),
    }


@torch.no_grad()
def evaluate(model, loader, sampler, *, device, world_size):
    model.eval()
    sampler.set_epoch(0)
    totals = {key: 0.0 for key in
              ("loss_sum", "total_sum", "router_sum", "tokens", "correct")}
    examples = 0.0
    by_length = {}
    for batch in loader:
        assert_same_length(batch, device)
        batch = {key: value.to(device, non_blocking=True)
                 for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            output = model(**batch)
        values = batch_metrics(output, batch["labels"], device)
        for key, value in values.items():
            totals[key] += value
        batch_examples = reduce_sum(batch["labels"].shape[0], device)
        examples += batch_examples
        length = int(batch["labels"].shape[1] + 1)
        length_totals = by_length.setdefault(
            length,
            {key: 0.0 for key in
             ("loss_sum", "router_sum", "tokens", "correct", "examples")},
        )
        for key in ("loss_sum", "router_sum", "tokens", "correct"):
            length_totals[key] += values[key]
        length_totals["examples"] += batch_examples
    model.train()
    tokens = max(1.0, totals["tokens"])
    loss = totals["loss_sum"] / tokens
    metrics_by_length = {}
    for length, length_totals in sorted(by_length.items()):
        length_tokens = max(1.0, length_totals["tokens"])
        length_loss = length_totals["loss_sum"] / length_tokens
        metrics_by_length[str(length)] = {
            "loss": length_loss,
            "next_token_loss": length_loss,
            "next_token_acc": length_totals["correct"] / length_tokens,
            "ppl": math.exp(min(20.0, length_loss)),
            "router": length_totals["router_sum"] / length_tokens,
            "examples": int(length_totals["examples"]),
            "tokens": int(length_totals["tokens"]),
        }
    return {
        "loss": loss,
        "next_token_loss": loss,
        "next_token_acc": totals["correct"] / tokens,
        "ppl": math.exp(min(20.0, loss)),
        "router": totals["router_sum"] / tokens,
        "examples": int(examples),
        "tokens": int(totals["tokens"]),
        "by_length": metrics_by_length,
    }


def load_model_weights(model, checkpoint_path: str, rank: int):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    raw_model = model.module if isinstance(model, DDP) else model
    current = raw_model.state_dict()
    compatible = {
        name: value for name, value in state.items()
        if name in current and current[name].shape == value.shape
    }
    raw_model.load_state_dict(compatible, strict=False)
    # Causal tuning variants may add a local residual projection.  Starting
    # it from the already-trained causal router projection gives the new path
    # a sensible representation without changing the old MLM checkpoint.
    if (getattr(raw_model, "local_skip", False)
            and "router.projection.weight" in compatible):
        with torch.no_grad():
            raw_model.local_skip_proj.weight.copy_(
                raw_model.router.projection.weight
            )
            raw_model.local_skip_proj.bias.copy_(
                raw_model.router.projection.bias
            )
    if rank == 0:
        print(json.dumps({
            "init_from": checkpoint_path,
            "loaded_model_tensors": len(compatible),
            "skipped_model_tensors": len(current) - len(compatible),
            "optimizer": "reset_for_next_token",
        }, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--init-from", required=True,
                        help="MLM/pre-training checkpoint used for model warm-start")
    parser.add_argument("--model-size", choices=("auto", *available_model_presets()),
                        default="auto",
                        help="architecture preset; auto reads the initialization checkpoint")
    parser.add_argument("--fasta", default="/gpfs_data/Projects/data/hg38/hg38.ml.fa")
    parser.add_argument("--bed", default="/gpfs_data/Projects/data/hg38/human-sequences.bed")
    parser.add_argument("--lengths", default=DEFAULT_LENGTHS)
    parser.add_argument("--max-length", type=int, default=131_072)
    parser.add_argument("--batch-sizes", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--total-updates", type=int, default=30_600)
    parser.add_argument("--warmup-updates", type=int, default=1_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--validation-samples", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir",
                        default=str(Path("weights/pretrain") /
                                     time.strftime("%Y%m%d") /
                                     "vericomp-next-token"))
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--causal-local-skip", action="store_true",
                        help="add a causal full-resolution residual path")
    parser.add_argument("--local-skip-gate-bias", type=float, default=-2.0)
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="causal bottleneck token budget")
    parser.add_argument("--no-fused-optimizer", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.grad_accumulation < 1:
        raise ValueError("--grad-accumulation must be positive")
    distributed, rank, world_size, device = setup_distributed()
    log_writer = AsyncLogWriter(rank == 0)
    try:
        lengths = [int(value) for value in args.lengths.split(",") if value.strip()]
        batch_sizes = parse_batch_sizes(args.batch_sizes)
        if any(length > args.max_length for length in lengths):
            raise ValueError("all --lengths must be <= --max-length")
        random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed_all(args.seed + rank)

        train_ds = BEDGenomeDataset(
            args.bed, args.fasta, args.max_length,
            sequence_lengths=lengths, split="train", seed=args.seed,
            random_window_sampling=True,
        )
        valid_ds = BEDGenomeDataset(
            args.bed, args.fasta, args.max_length,
            sequence_lengths=lengths, split="valid",
            max_samples=args.validation_samples, seed=args.seed + 10_000,
        )
        train_loader, train_sampler = make_loader(
            train_ds, world_size=world_size, rank=rank,
            batch_sizes=batch_sizes, seed=args.seed, shuffle=True,
            drop_last=True, workers=args.num_workers,
        )
        valid_loader, valid_sampler = make_loader(
            valid_ds, world_size=world_size, rank=rank,
            batch_sizes=batch_sizes, seed=args.seed + 1, shuffle=False,
            drop_last=False, workers=args.num_workers,
        )

        model_config = model_config_from_checkpoint(
            args.init_from,
            args.model_size,
            activation_checkpointing=args.activation_checkpointing,
            causal_attention=True,
            max_tokens=args.max_tokens,
        )
        model = CausalVeriCompNextTokenModel(
            model_config,
            local_skip=args.causal_local_skip,
            local_skip_gate_bias=args.local_skip_gate_bias,
        ).to(device)
        if distributed:
            model = DDP(model, device_ids=[device.index], output_device=device.index,
                        gradient_as_bucket_view=True)
        load_model_weights(model, args.init_from, rank)
        optimizer = build_optimizer(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            fused=not args.no_fused_optimizer,
        )
        scheduler = build_scheduler(optimizer, args.warmup_updates,
                                    args.total_updates)
        output_dir = Path(args.output_dir)
        if rank == 0:
            runtime_metadata = {
                "world_size": world_size,
                "device": torch.cuda.get_device_name(device),
                "parameters": (model.module if isinstance(model, DDP) else model).num_parameters(),
                "resolved_lengths": lengths,
                "per_rank_batch_sizes": batch_sizes,
                "global_batch_sizes": {length: batch * world_size
                                        for length, batch in batch_sizes.items()},
                "train_samples": len(train_ds),
                "validation_samples": len(valid_ds),
                "optimizer": "fused_adamw" if not args.no_fused_optimizer else "adamw",
                "resume_mode": "warm_start_from_init_from",
                "torch_version": torch.__version__,
            }
            print_resolved_config(
                entrypoint="pretrain.next_token", args=args,
                model_config=model_config, runtime=runtime_metadata,
                extra={"data": {"fasta": args.fasta, "bed": args.bed}},
            )
            write_run_config(
                output_dir,
                entrypoint="pretrain.next_token",
                model_path=Path(__file__).resolve().parents[1] / "vericomp/causal_modeling.py",
                model_class="CausalVeriCompNextTokenModel",
                model_config=model_config,
                args=args,
                runtime=runtime_metadata,
                extra={"data": {"fasta": args.fasta, "bed": args.bed}},
            )
        best_path = output_dir / "best.pt"
        best_val_loss = float("inf")

        if rank == 0:
            print(json.dumps({
                "task": "next_token",
                "causal_attention": True,
                "causal_local_skip": args.causal_local_skip,
                "local_skip_gate_bias": args.local_skip_gate_bias,
                "max_tokens": args.max_tokens,
                "train_examples": len(train_ds),
                "validation_examples": len(valid_ds),
                "lengths": lengths,
                "batch_sizes": batch_sizes,
                "total_updates": args.total_updates,
                "output_dir": str(output_dir),
            }, indent=2), flush=True)

        train_sampler.set_epoch(0)
        iterator = iter(train_loader)
        update = 0
        epoch = 0
        last_log = time.perf_counter()
        start_time = last_log
        measured_updates = 0
        eta_start = None
        total_tokens_seen = 0

        while update < args.total_updates:
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            step_loss_sum = step_tokens = step_correct = 0.0
            last_length = 0
            last_global_batch = 0
            for micro_step in range(args.grad_accumulation):
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch += 1
                    train_sampler.set_epoch(epoch)
                    iterator = iter(train_loader)
                    batch = next(iterator)
                assert_same_length(batch, device)
                batch = {key: value.to(device, non_blocking=True)
                         for key, value in batch.items()}
                last_length = batch["input_ids_pos"].shape[1]
                last_global_batch = batch["input_ids_pos"].shape[0] * world_size
                with (model.no_sync() if isinstance(model, DDP)
                      and micro_step < args.grad_accumulation - 1
                      else contextlib.nullcontext()):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                        enabled=device.type == "cuda"):
                        output = model(**batch)
                        loss = output["loss"] / args.grad_accumulation
                    loss.backward()
                valid = batch["labels"].ne(-100)
                valid_tokens = float(valid.sum().item())
                step_loss_sum += float(output["loss_mlm"].detach().float()) * valid_tokens
                step_tokens += valid_tokens
                step_correct += float((output["logits"].argmax(-1).eq(batch["labels"]) & valid).sum())

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            update += 1
            elapsed = max(1e-9, time.perf_counter() - step_start)
            total_tokens_seen += int(step_tokens * world_size)
            measured_updates += 1
            if update == 1:
                eta_start = time.perf_counter()

            if update % args.eval_every == 0 or update == args.total_updates:
                metrics = evaluate(model, valid_loader, valid_sampler,
                                   device=device, world_size=world_size)
                if rank == 0:
                    print(json.dumps({"validation_update": update,
                                      **metrics}), flush=True)
                    if metrics["loss"] < best_val_loss:
                        best_val_loss = metrics["loss"]
                        save_checkpoint(
                            best_path, model, optimizer, scheduler,
                            update=update, epoch=epoch, args=args,
                            best_val_loss=best_val_loss,
                        )

            if rank == 0 and (update % args.save_every == 0
                              or update == args.total_updates):
                save_checkpoint(
                    output_dir / "latest.pt", model, optimizer, scheduler,
                    update=update, epoch=epoch, args=args,
                    best_val_loss=best_val_loss,
                )

            if rank == 0 and (update == 1 or update % 10 == 0
                              or update == args.total_updates):
                now = time.perf_counter()
                interval = max(1e-9, now - last_log)
                last_log = now
                if eta_start is not None and measured_updates > 1:
                    seconds_per_update = (now - eta_start) / (measured_updates - 1)
                    eta = format_duration((args.total_updates - update) * seconds_per_update)
                else:
                    eta = "warming_up"
                log_writer.log(
                    f"update={update}/{args.total_updates} L={last_length} "
                    f"local_batch={last_global_batch // world_size} "
                    f"global_batch={last_global_batch} "
                    f"lr={scheduler.get_last_lr()[0]:.6g} "
                    f"next_token_loss={step_loss_sum / max(1.0, step_tokens):.5f} "
                    f"ppl={math.exp(min(20.0, step_loss_sum / max(1.0, step_tokens))):.4f} "
                    f"next_token_acc={step_correct / max(1.0, step_tokens):.5f} "
                    f"grad_norm={float(grad_norm):.4f} step_s={elapsed:.2f} "
                    f"interval_s={interval:.2f} tokens_seen={total_tokens_seen} "
                    f"eta={eta}"
                )
        if distributed:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        log_writer.close()


if __name__ == "__main__":
    main()
