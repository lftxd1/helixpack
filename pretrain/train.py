"""Four-GPU mixed-length BF16 DDP pre-training for VeriComp.

The production defaults are the requested run:

* 4 GPUs / DDP
* length-dependent per-rank batch sizes
* gradient accumulation 1
* 24,414 optimizer updates
* linear warmup + linear decay, peak LR 3e-4, warmup 1000 updates
* CUDA fused AdamW, BF16 autocast, global gradient norm clip 2
* up to 300 validation examples

Every DDP step receives one same-length global batch. The next optimizer step
may use a different length. The sampler splits the global batch across ranks,
so no rank can get a different sequence length at the same step.

Launch:

  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m pretrain.train
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import random
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from vericomp import VeriCompConfig, VeriCompPretrainModel
from pretrain.run_config import print_resolved_config, write_run_config
from vericomp.data import BEDGenomeDataset, DistributedLengthBatchSampler, SpanMLMCollator
from vericomp.model_presets import available_model_presets, model_config_from_preset


DEFAULT_LENGTHS = "1024,2048,4096,8192,16384,32768,65536,131072"
DEFAULT_BATCH_SIZES = "1024:128,2048:64,4096:32,8192:32,16384:16,32768:16,65536:8,131072:8"


class AsyncLogWriter:
    """Write rank-zero logs from a background thread.

    ``log`` only enqueues a string and never waits for the file/terminal
    writer.  The writer flushes every message, which is important for nohup
    jobs where the log is monitored while training is running.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._closed = False
        self._queue = queue.SimpleQueue()
        self._thread = None
        if enabled:
            self._thread = threading.Thread(
                target=self._run, name="vericomp-log-writer", daemon=True
            )
            self._thread.start()

    def _run(self):
        while True:
            message = self._queue.get()
            if message is None:
                return
            sys.stdout.write(message)
            if not message.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()

    def log(self, message: str):
        if self.enabled and not self._closed:
            self._queue.put(str(message))

    def close(self):
        if not self.enabled or self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {seconds:02d}s"


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, torch.device("cuda", local_rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, device


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--fasta", default="/gpfs_data/Projects/data/hg38/hg38.ml.fa")
    p.add_argument("--bed", default="/gpfs_data/Projects/data/hg38/human-sequences.bed")
    p.add_argument("--lengths", default=DEFAULT_LENGTHS,
                   help="comma-separated exact lengths used as separate buckets")
    p.add_argument("--max-length", type=int, default=131_072,
                   help="training cap; inference remains capable of 1M")
    p.add_argument("--batch-sizes", default=DEFAULT_BATCH_SIZES,
                   help="length:per-GPU-batch mapping, comma-separated")
    p.add_argument("--batch-size", type=int, default=None,
                   help="legacy uniform per-GPU batch size; overrides --batch-sizes")
    p.add_argument("--grad-accumulation", type=int, default=1)
    p.add_argument("--total-updates", type=int, default=24_414)
    p.add_argument("--warmup-updates", type=int, default=1_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-end", type=float, default=None,
                   help="final learning rate for linear decay; defaults to zero")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=2.0)
    p.add_argument("--validation-samples", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=500,
                   help="validate and checkpoint every N optimizer updates")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--output-dir",
                   default=str(Path("weights/pretrain") / time.strftime("%Y%m%d")
                                / "vericomp-128k"))
    p.add_argument("--resume", default=None)
    p.add_argument("--warm-start", action="store_true",
                   help="load --resume model weights only and reset optimizer/scheduler/update")
    p.add_argument("--model-size", choices=available_model_presets(), default="100m",
                   help="named architecture preset; defaults to the historical 100M model")
    p.add_argument("--num-layers", type=int, default=None,
                   help="optional architecture override retained for legacy experiments")
    p.add_argument("--no-activation-checkpointing", action="store_true",
                   help="disable memory-saving checkpointing; not recommended at 128K")
    p.add_argument("--no-fused-optimizer", action="store_true",
                   help="disable CUDA fused AdamW (for compatibility/debugging)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--allow-non4", action="store_true",
                   help="allow a smoke test with a world size other than 4")
    return p.parse_args()


def parse_batch_sizes(spec: str) -> dict[int, int]:
    """Parse a comma-separated ``length:local_batch`` mapping."""
    result = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            length_text, batch_text = item.split(":", 1)
            length, batch = int(length_text), int(batch_text)
        except ValueError as exc:
            raise ValueError(
                f"invalid --batch-sizes item {item!r}; expected length:batch"
            ) from exc
        if length <= 0 or batch <= 0:
            raise ValueError("batch-sizes lengths and batch sizes must be positive")
        if length in result:
            raise ValueError(f"duplicate batch size for length {length}")
        result[length] = batch
    if not result:
        raise ValueError("--batch-sizes must contain at least one length:batch pair")
    return result


def assert_same_length(batch, device):
    """Fail before the forward pass if ranks have diverged in sequence length."""
    if not dist.is_initialized():
        return
    value = torch.tensor([batch["input_ids_pos"].shape[1]], device=device, dtype=torch.int64)
    low, high = value.clone(), value.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN)
    dist.all_reduce(high, op=dist.ReduceOp.MAX)
    if low.item() != high.item():
        raise RuntimeError(f"DDP ranks received lengths {low.item()} and {high.item()}")


def make_loader(dataset, *, world_size, rank, batch_sizes=None, batch_size=None,
                seed, shuffle, drop_last, workers, equal_length_sampling=False):
    if batch_sizes is None:
        if batch_size is None:
            raise ValueError("one of batch_sizes or batch_size must be provided")
        batch_sizes = batch_size
    sampler = DistributedLengthBatchSampler(
        dataset,
        world_size=world_size,
        rank=rank,
        per_rank_batch_size=batch_sizes,
        drop_last=drop_last,
        shuffle=shuffle,
        seed=seed,
        equal_length_sampling=equal_length_sampling,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=SpanMLMCollator(),
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    return loader, sampler


def build_scheduler(optimizer, warmup_updates, total_updates, lr_end=None):
    start_lr = float(optimizer.param_groups[0]["lr"])
    end_lr = 0.0 if lr_end is None else float(lr_end)
    if start_lr <= 0 or end_lr < 0 or end_lr > start_lr:
        raise ValueError("lr-end must satisfy 0 <= lr-end <= lr")

    def lr_lambda(step):
        if step < warmup_updates:
            return float(step + 1) / max(1, warmup_updates)
        remaining = max(1, total_updates - warmup_updates)
        progress = min(1.0, max(0.0, float(step - warmup_updates) / remaining))
        return (start_lr + (end_lr - start_lr) * progress) / start_lr
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_optimizer(parameters, *, lr: float, weight_decay: float,
                    fused: bool = True):
    """Build the fastest compatible AdamW for the current device.

    PyTorch's fused CUDA implementation updates all parameter tensors from one
    fused path, which is important here because the model has many MoE expert
    tensors.  Keep a portable fallback for CPU/smoke-test environments.
    """
    kwargs = {"lr": lr, "weight_decay": weight_decay}
    use_fused = bool(fused and torch.cuda.is_available())
    if use_fused:
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(parameters, **kwargs)
    except (TypeError, RuntimeError):
        if not use_fused:
            raise
        kwargs.pop("fused", None)
        return torch.optim.AdamW(parameters, **kwargs)


def reduce_weighted(value: torch.Tensor, local_count: int):
    """Return global sum and global sample count for a local mean metric."""
    value = value.detach().float() * local_count
    count = torch.tensor(float(local_count), device=value.device)
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return value.item(), count.item()


def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor):
    """Return correct masked-token count and masked-token count."""
    mask = labels.ne(-100)
    count = mask.sum().to(torch.float32)
    correct = (logits.argmax(dim=-1).eq(labels) & mask).sum().to(torch.float32)
    if dist.is_initialized():
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return correct.item(), count.item()


@torch.no_grad()
def evaluate(model, loader, sampler, *, device, world_size, max_batches=None):
    model.eval()
    sampler.set_epoch(0)
    total_loss = 0.0
    total_mlm = 0.0
    total_strand = 0.0
    total_router = 0.0
    total_correct = 0.0
    total_masked = 0.0
    count = 0.0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        assert_same_length(batch, device)
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            out = model(**batch)
        local_count = batch["input_ids_pos"].shape[0]
        loss_sum, global_count = reduce_weighted(out["loss"], local_count)
        mlm_sum, _ = reduce_weighted(out["loss_mlm"], local_count)
        strand_sum, _ = reduce_weighted(out["loss_strand"], local_count)
        router_sum, _ = reduce_weighted(out["loss_router"], local_count)
        correct, masked = masked_accuracy(out["logits"], batch["labels"])
        total_loss += loss_sum
        total_mlm += mlm_sum
        total_strand += strand_sum
        total_router += router_sum
        total_correct += correct
        total_masked += masked
        count += global_count
    model.train()
    if count == 0:
        return {"loss": float("nan"), "loss_mlm": float("nan"),
                "loss_strand": float("nan"), "loss_router": float("nan"),
                "mlm_acc": float("nan"), "examples": 0}
    return {"loss": total_loss / count, "loss_mlm": total_mlm / count,
            "loss_strand": total_strand / count,
            "loss_router": total_router / count,
            "mlm_acc": total_correct / max(1.0, total_masked),
            "examples": int(count)}


def save_checkpoint(path, model, optimizer, scheduler, *, update, epoch, args,
                    best_val_loss=None):
    raw_model = model.module if isinstance(model, DDP) else model
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "update": update,
        "epoch": epoch,
        "args": vars(args),
    }
    model_config = getattr(raw_model, "config", None)
    if model_config is not None:
        checkpoint["model_config"] = asdict(model_config)
    if best_val_loss is not None:
        checkpoint["best_val_loss"] = best_val_loss
    torch.save(checkpoint, path)


def main():
    args = parse_args()
    distributed, rank, world_size, device = setup_distributed()
    log_writer = AsyncLogWriter(rank == 0)
    try:
        if world_size != 4 and not args.allow_non4:
            raise RuntimeError(
                f"This training recipe requires four GPUs; got WORLD_SIZE={world_size}. "
                "Use torchrun --nproc_per_node=4, or --allow-non4 for a smoke test."
            )
        if device.type != "cuda":
            raise RuntimeError("The requested BF16 + Flash Attention training requires CUDA")
        if not torch.backends.cuda.is_flash_attention_available():
            raise RuntimeError("PyTorch Flash Attention backend is unavailable")
        if args.grad_accumulation < 1 or args.total_updates < 1:
            raise ValueError("grad accumulation and total updates must be positive")
        if args.warmup_updates > args.total_updates:
            raise ValueError("warmup updates cannot exceed total updates")

        torch.manual_seed(args.seed + rank)
        random.seed(args.seed + rank)
        torch.cuda.manual_seed_all(args.seed + rank)
        lengths = [int(value) for value in args.lengths.split(",") if value.strip()]
        if not lengths or any(length <= 0 or length > args.max_length for length in lengths):
            raise ValueError("all --lengths must be in (0, --max-length]")
        batch_sizes = parse_batch_sizes(args.batch_sizes)
        if args.batch_size is not None:
            batch_sizes = {length: args.batch_size for length in lengths}
        missing_batch_sizes = sorted(set(lengths) - set(batch_sizes))
        if missing_batch_sizes:
            raise ValueError(
                f"no per-rank batch size configured for lengths {missing_batch_sizes}; "
                "add them to --batch-sizes"
            )

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
            batch_sizes=batch_sizes, seed=args.seed,
            shuffle=True, drop_last=True, workers=args.num_workers,
            equal_length_sampling=True,
        )
        valid_loader, valid_sampler = make_loader(
            valid_ds, world_size=world_size, rank=rank,
            batch_sizes=batch_sizes, seed=args.seed + 1,
            # Every DDP rank must iterate the same number of validation
            # batches.  Keeping per-bucket tails would split an incomplete
            # global batch unevenly, causing ranks to become misaligned on
            # the next collective (and potentially receive different L).
            shuffle=False, drop_last=True, workers=args.num_workers,
            equal_length_sampling=False,
        )

        config_overrides = {
            "activation_checkpointing": not args.no_activation_checkpointing,
        }
        if args.num_layers is not None:
            config_overrides["num_layers"] = args.num_layers
        model_config = model_config_from_preset(args.model_size, **config_overrides)
        model = VeriCompPretrainModel(model_config).to(device)
        if distributed:
            model = DDP(model, device_ids=[device.index], output_device=device.index,
                        gradient_as_bucket_view=True)
        optimizer = build_optimizer(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            fused=not args.no_fused_optimizer,
        )
        scheduler = build_scheduler(
            optimizer, args.warmup_updates, args.total_updates, args.lr_end
        )
        start_update, epoch = 0, 0
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            raw_model = model.module if isinstance(model, DDP) else model
            checkpoint_state = checkpoint["model"]
            current_state = raw_model.state_dict()
            compatible_state = {
                name: value for name, value in checkpoint_state.items()
                if name in current_state and current_state[name].shape == value.shape
            }
            missing_names = sorted(set(current_state) - set(compatible_state))
            unexpected_names = sorted(set(checkpoint_state) - set(compatible_state))
            raw_model.load_state_dict(compatible_state, strict=False)
            if not missing_names and not unexpected_names and not args.warm_start:
                optimizer.load_state_dict(checkpoint["optimizer"])
                scheduler.load_state_dict(checkpoint["scheduler"])
                start_update = int(checkpoint["update"])
                epoch = int(checkpoint.get("epoch", 0))
                resume_mode = "exact"
            else:
                # New residual-skip parameters have no optimizer slots in an
                # older checkpoint.  Warm-start matching weights and restart
                # optimizer/scheduler state rather than silently misaligning
                # parameter groups.
                resume_mode = "warm_start"
                if rank == 0:
                    print(
                        f"resume={args.resume}: warm-started {len(compatible_state)} "
                        f"matching tensors; skipped {len(missing_names)} new/mismatched "
                        "tensors and reset optimizer/scheduler",
                        flush=True,
                    )

        raw_model = model.module if isinstance(model, DDP) else model
        if rank == 0:
            runtime_metadata = {
                "world_size": world_size,
                "device": torch.cuda.get_device_name(device),
                "parameters": raw_model.num_parameters(),
                "per_rank_batch_sizes": batch_sizes,
                "global_batch_sizes": {length: batch * world_size
                                        for length, batch in batch_sizes.items()},
                "grad_accumulation": args.grad_accumulation,
                "effective_global_samples_per_update": "length-dependent",
                "total_updates": args.total_updates,
                "warmup_updates": args.warmup_updates,
                "peak_lr": args.lr,
                "final_lr": args.lr if args.lr_end is None else args.lr_end,
                "lengths": lengths,
                "random_window_sampling": train_ds.random_window_sampling,
                "equal_length_sampling": train_sampler.equal_length_sampling,
                "train_samples": len(train_ds),
                "validation_samples_requested": args.validation_samples,
                "validation_samples_available": len(valid_ds),
                "optimizer": "fused_adamw" if not args.no_fused_optimizer else "adamw",
                "strand_loss_backward": False,
                "resume_mode": locals().get("resume_mode", "fresh"),
                "torch_version": torch.__version__,
            }
            print_resolved_config(
                entrypoint="pretrain.train", args=args, model_config=model_config,
                runtime=runtime_metadata,
                extra={"data": {"fasta": args.fasta, "bed": args.bed}},
            )
            log_writer.log(json.dumps({
                **runtime_metadata,
            }, indent=2))

        output_dir = Path(args.output_dir)
        if rank == 0:
            write_run_config(
                output_dir,
                entrypoint="pretrain.train",
                model_path=Path(__file__).resolve().parents[1] / "vericomp/modeling.py",
                model_class="VeriCompPretrainModel",
                model_config=model_config,
                args=args,
                runtime={**runtime_metadata, "resolved_lengths": lengths},
                extra={"data": {"fasta": args.fasta, "bed": args.bed}},
            )
        best_val_loss = float("inf")
        best_path = output_dir / "best.pt"
        if best_path.exists():
            try:
                previous_best = torch.load(best_path, map_location="cpu",
                                           weights_only=False)
                best_val_loss = float(previous_best.get("best_val_loss", "inf"))
            except (OSError, RuntimeError, ValueError, TypeError):
                best_val_loss = float("inf")
        iterator = None
        train_sampler.set_epoch(epoch)
        iterator = iter(train_loader)
        update = start_update
        last_log = time.perf_counter()
        eta_start = None
        ema_step_seconds = None
        measured_updates = 0
        total_samples_seen = 0
        total_tokens_seen = 0
        total_compute_seconds = 0.0
        while update < args.total_updates:
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            last_length = None
            update_samples = 0
            update_tokens = 0
            last_global_batch = 0
            loss_for_log = mlm_for_log = strand_for_log = router_for_log = 0.0
            correct_for_log = masked_for_log = 0.0
            for micro_step in range(args.grad_accumulation):
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch += 1
                    train_sampler.set_epoch(epoch)
                    iterator = iter(train_loader)
                    batch = next(iterator)
                assert_same_length(batch, device)
                last_length = batch["input_ids_pos"].shape[1]
                global_samples = batch["input_ids_pos"].shape[0] * world_size
                last_global_batch = global_samples
                update_samples += global_samples
                update_tokens += global_samples * last_length
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                sync_context = (
                    model.no_sync()
                    if isinstance(model, DDP) and micro_step < args.grad_accumulation - 1
                    else contextlib.nullcontext()
                )
                with sync_context:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = model(**batch)
                        loss = out["loss"] / args.grad_accumulation
                    loss.backward()
                loss_for_log += out["loss"].detach().float().item()
                mlm_for_log += out["loss_mlm"].detach().float().item()
                strand_for_log += out["loss_strand"].detach().float().item()
                router_for_log += out["loss_router"].detach().float().item()
                local_correct, local_masked = masked_accuracy(out["logits"].detach(), batch["labels"])
                correct_for_log += local_correct
                masked_for_log += local_masked

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            update += 1
            step_seconds = max(1e-9, time.perf_counter() - step_start)
            total_compute_seconds += step_seconds
            total_samples_seen += update_samples
            total_tokens_seen += update_tokens
            if update == start_update + 1:
                # Exclude first-step compilation/autotuning from ETA once the
                # second completed update is available.
                eta_start = time.perf_counter()
            else:
                measured_updates += 1
                if ema_step_seconds is None:
                    ema_step_seconds = step_seconds
                else:
                    ema_step_seconds = 0.9 * ema_step_seconds + 0.1 * step_seconds

            if update % args.eval_every == 0 or update == args.total_updates:
                metrics = evaluate(model, valid_loader, valid_sampler,
                                   device=device, world_size=world_size)
                if rank == 0:
                    print(f"validation update={update} examples={metrics['examples']} "
                          f"loss={metrics['loss']:.5f} mlm={metrics['loss_mlm']:.5f} "
                                 f"mlm_acc={metrics['mlm_acc']:.5f} "
                                 f"strand={metrics['loss_strand']:.6f} "
                                 f"router={metrics['loss_router']:.6f}", flush=True)
                    if (metrics["loss"] == metrics["loss"]
                            and metrics["loss"] < best_val_loss):
                        best_val_loss = metrics["loss"]
                        save_checkpoint(
                            best_path, model, optimizer, scheduler,
                            update=update, epoch=epoch, args=args,
                            best_val_loss=best_val_loss,
                        )
            if rank == 0 and (update % args.save_every == 0 or update == args.total_updates):
                save_checkpoint(output_dir / "latest.pt", model, optimizer, scheduler,
                                update=update, epoch=epoch, args=args,
                                best_val_loss=best_val_loss)

            if rank == 0 and (update == 1 or update % 10 == 0 or update == args.total_updates):
                now = time.perf_counter()
                interval_seconds = max(1e-9, now - last_log)
                last_log = now
                current_samples_s = update_samples / step_seconds
                current_tokens_s = update_tokens / step_seconds
                average_tokens_s = total_tokens_seen / max(total_compute_seconds, 1e-9)
                if eta_start is not None and measured_updates > 0:
                    eta_step = max(0.0, (now - eta_start) / measured_updates)
                    remaining = max(0, args.total_updates - update)
                    eta_text = format_duration(remaining * eta_step)
                else:
                    eta_text = "warming_up"
                log_writer.log(
                    f"update={update}/{args.total_updates} L={last_length} "
                    f"local_batch={last_global_batch // world_size} "
                    f"global_batch={last_global_batch} "
                    f"lr={scheduler.get_last_lr()[0]:.6g} "
                    f"loss={loss_for_log / args.grad_accumulation:.5f} "
                    f"mlm={mlm_for_log / args.grad_accumulation:.5f} "
                    f"mlm_acc={correct_for_log / max(1.0, masked_for_log):.5f} "
                    f"strand={strand_for_log / args.grad_accumulation:.6f} "
                    f"router={router_for_log / args.grad_accumulation:.6f} "
                    f"grad_norm={float(grad_norm):.4f} "
                    f"step_s={step_seconds:.2f} interval_s={interval_seconds:.2f} "
                    f"throughput_samples_s={current_samples_s:.2f} "
                    f"throughput_tokens_s={current_tokens_s:.2f} "
                    f"avg_tokens_s={average_tokens_s:.2f} "
                    f"tokens_seen={total_tokens_seen} eta={eta_text}"
                )
        if distributed:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        log_writer.close()


if __name__ == "__main__":
    main()
