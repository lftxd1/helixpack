"""Persist reproducibility metadata next to every pretraining run."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def print_resolved_config(*, entrypoint: str, args, model_config=None,
                          runtime: dict | None = None,
                          extra: dict | None = None) -> None:
    """Print the complete resolved pretraining recipe before training starts."""
    payload = {
        "event": "resolved_config_before_training",
        "entrypoint": entrypoint,
        "arguments": vars(args),
    }
    if model_config is not None:
        payload["model_config"] = model_config
    if runtime:
        payload["runtime"] = runtime
    if extra:
        payload.update(extra)
    print(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), flush=True)


def write_run_config(output_dir: Path, *, entrypoint: str, model_path: Path,
                     model_class: str, model_config, args, runtime: dict,
                     extra: dict | None = None) -> Path:
    """Write ``run_config.json`` once, preserving the original run recipe."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_config.json"
    if path.exists():
        return path
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "entrypoint": entrypoint,
        "command": [str(item) for item in sys.argv],
        "model": {
            "implementation_path": str(model_path.resolve()),
            "class": model_class,
            "config": _jsonable(model_config),
        },
        "training": _jsonable(vars(args)),
        "runtime": _jsonable(runtime),
    }
    if extra:
        payload.update(_jsonable(extra))
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path
