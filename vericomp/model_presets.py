"""Named model configurations used by training and downstream jobs.

The 100M preset intentionally matches ``VeriCompConfig()``.  Keeping the
default dataclass unchanged is important for backwards compatibility with
existing commands, checkpoints, and experiment scripts.
"""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

from .config import VeriCompConfig


# These fields determine tensor shapes.  Runtime options such as dropout,
# activation checkpointing, and strict short-sequence handling may legitimately
# be overridden by a downstream job and therefore are not included here.
ARCHITECTURE_FIELDS = (
    "vocab_size",
    "output_vocab_size",
    "shallow_dim",
    "shallow_mamba_layers",
    "shallow_mamba_state",
    "shallow_mamba_conv",
    "shallow_mamba_expand",
    "shallow_mamba_head_dim",
    "d_model",
    "num_layers",
    "nhead",
    "dim_feedforward",
    "moe_intermediate_size",
    "num_experts",
    "num_experts_per_tok",
    "expert_layer_period",
    "expert_layer_offset",
    "max_tokens",
    "max_position_embeddings",
)


# These fields do not change tensor shapes, but they change which forward
# path a short input takes.  ``strict_uncompressed_short`` is the load-bearing
# one: the long-context recipe trains with it enabled, so inputs shorter than
# ``max_tokens`` bypass allocation entirely.  Downstream jobs must reproduce
# that, otherwise a short sequence is routed through a dynamic allocation the
# checkpoint never saw.  Values are recovered from a legacy checkpoint's
# pretraining ``args`` unless the caller overrides them explicitly.
FORWARD_SHAPING_FIELDS = (
    "allocation_mode",
    "strict_uncompressed_short",
    "duplex_mode",
)

_FORWARD_SHAPING_DEFAULTS: dict[str, Any] = {}


MODEL_PRESETS = {
    "100m": {},
    # Approximately 450M parameters: keep the 64-dimensional attention head
    # used by the 100M model while scaling depth and width together.  The
    # shallow encoder, tokenizer, bottleneck budget, and decoder interface
    # remain unchanged so this is a new checkpoint family, not a replacement
    # for the historical 100M default.
    "450m": {
        "d_model": 768,
        "num_layers": 24,
        "nhead": 12,
        "dim_feedforward": 10752,
        "moe_intermediate_size": 424,
    },
    # The explicit layers are scaled while the shallow encoder and the
    # bottleneck budget remain unchanged.  With 16 MoE and 16 dense layers,
    # this is approximately a 1B-parameter configuration for the current
    # architecture; the runtime parameter count is always recorded and must
    # be checked before a production run.
    "1b": {
        "d_model": 1024,
        "num_layers": 32,
        "nhead": 16,
        "dim_feedforward": 13312,
        "moe_intermediate_size": 544,
    },
}


def available_model_presets() -> tuple[str, ...]:
    return tuple(MODEL_PRESETS)


def architecture_mismatches(checkpoint_config: dict[str, Any],
                            current_config: VeriCompConfig) -> dict[str, tuple[Any, Any]]:
    """Return shape-relevant config differences between two model configs."""
    return {
        field: (checkpoint_config[field], getattr(current_config, field))
        for field in ARCHITECTURE_FIELDS
        if field in checkpoint_config
        and checkpoint_config[field] != getattr(current_config, field)
    }


def model_config_from_preset(name: str = "100m", **overrides: Any) -> VeriCompConfig:
    """Build a config from a named preset without mutating global defaults."""
    normalized = str(name).strip().lower()
    if normalized not in MODEL_PRESETS:
        available = ", ".join(available_model_presets())
        raise ValueError(f"unknown model preset {name!r}; expected one of {available}")
    config = VeriCompConfig(**MODEL_PRESETS[normalized])
    if overrides:
        valid_fields = {field.name for field in fields(VeriCompConfig)}
        unknown = sorted(set(overrides) - valid_fields)
        if unknown:
            raise TypeError(f"unknown VeriCompConfig fields: {', '.join(unknown)}")
        config = replace(config, **overrides)
    return config


def model_config_from_checkpoint(
    checkpoint_path: str,
    model_size: str = "auto",
    **overrides: Any,
) -> VeriCompConfig:
    """Resolve a model config from checkpoint metadata or an explicit preset.

    Checkpoints written by the current trainer carry ``model_config``.  Older
    ones do not, but they still store the pretraining ``args``, which record
    the forward-shaping fields (``allocation_mode``, ``strict_uncompressed_short``,
    ``duplex_mode``).  Those are recovered here because they decide whether a
    short input bypasses allocation.  Recovering them keeps a downstream job on
    the same forward path the checkpoint was trained with; without it, short
    inputs fall through to the dataclass default and take a dynamic allocation
    path that pretraining never used.  Architecture still comes from the
    historical 100M preset for these legacy checkpoints.
    """
    normalized = str(model_size).strip().lower()
    if normalized not in {"auto", *MODEL_PRESETS}:
        available = ", ".join(("auto", *available_model_presets()))
        raise ValueError(f"unknown model size {model_size!r}; expected one of {available}")

    if normalized == "auto":
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            payload = checkpoint.get("model_config")
            if isinstance(payload, dict):
                valid_fields = {field.name for field in fields(VeriCompConfig)}
                values = {key: value for key, value in payload.items() if key in valid_fields}
                values.update(overrides)
                return replace(VeriCompConfig(), **values)

            recovered: dict[str, Any] = {}
            pretrain_args = checkpoint.get("args")
            if isinstance(pretrain_args, dict):
                for field in FORWARD_SHAPING_FIELDS:
                    if field in pretrain_args and pretrain_args[field] is not None:
                        recovered[field] = pretrain_args[field]
            recovered.update(overrides)
            normalized = "100m"
            return model_config_from_preset(normalized, **recovered)

        normalized = "100m"

    return model_config_from_preset(normalized, **overrides)
