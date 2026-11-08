"""Persist per-subtask fine-tuning metadata."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path


def resolve_model_path(model_spec: str) -> str:
    spec = str(model_spec).strip()
    path_text = spec.split(":", 1)[0]
    path = Path(path_text).expanduser()
    if path.suffix == ".py" or path.is_file():
        return str(path.resolve())
    aliases = {
        "vericomp": "vericomp.modeling",
        "mlm": "vericomp.modeling",
        "vericomp_mlm": "vericomp.modeling",
        "causal": "vericomp.causal_modeling",
        "causal_local_skip": "vericomp.causal_modeling",
        "vericomp_causal": "vericomp.causal_modeling",
    }
    module_name = aliases.get(spec, spec.split(":", 1)[0])
    try:
        module = import_module(module_name)
        module_path = getattr(module, "__file__", None)
        return str(Path(module_path).resolve()) if module_path else spec
    except (ImportError, ModuleNotFoundError):
        return spec


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def print_resolved_config(*, task: str, subtask: str, args,
                          model_path: str | None = None,
                          model_config=None, metadata: dict | None = None) -> None:
    """Print the complete resolved fine-tuning recipe before training starts."""
    payload = {
        "event": "resolved_config_before_training",
        "task": task,
        "subtask": subtask,
        "arguments": vars(args),
    }
    if model_path is not None:
        payload["model_path"] = model_path
    if model_config is not None:
        payload["model_config"] = model_config
    if metadata:
        payload.update(metadata)
    print(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), flush=True)


def write_finetune_config(output_dir: Path, *, task: str, subtask: str,
                          args, model_path: str, model_config=None,
                          extra: dict | None = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_config.json"
    if path.exists():
        return path
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "entrypoint": str(sys.argv[0]),
        "command": [str(item) for item in sys.argv],
        "task": task,
        "subtask": subtask,
        "model_path": model_path,
        "model_spec": str(getattr(args, "model", "")),
        "feature": str(getattr(args, "feature", "")),
        "checkpoint_path": str(Path(args.checkpoint).expanduser().resolve()),
        "parameters": _jsonable(vars(args)),
    }
    if model_config is not None:
        payload["model_config"] = _jsonable(model_config)
    if extra:
        payload.update(_jsonable(extra))
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path
