"""Backbone registry used by downstream fine-tuning jobs.

Built-in names are intentionally small and explicit.  A custom compatible
backbone can also be supplied as ``python.module:ClassName``; it must accept a
``VeriCompConfig`` and support ``forward(input_ids_pos=..., return_bottleneck=True)``
with ``bottleneck`` and ``loss_router`` entries in its output dictionary.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
from importlib import import_module
from pathlib import Path

from torch import nn

from .causal_modeling import CausalVeriCompNextTokenModel
from .config import VeriCompConfig
from .modeling import VeriCompPretrainModel


_COMPLEMENT_IDS = None


def reverse_complement_ids(input_ids):
    """Build the reverse-complement token sequence from A/C/G/T/N ids."""
    global _COMPLEMENT_IDS
    if _COMPLEMENT_IDS is None or _COMPLEMENT_IDS.device != input_ids.device:
        _COMPLEMENT_IDS = input_ids.new_tensor([3, 2, 1, 0, 4, 5])
    return _COMPLEMENT_IDS[input_ids.flip(dims=[-1])]


def extract_feature(backbone, input_ids, feature: str, *, dual_strand: bool = False):
    """Run a compatible backbone and return a ``[B,N,D]`` feature stream."""
    feature = str(feature).strip().lower()
    input_ids_rc = reverse_complement_ids(input_ids) if dual_strand else None
    if feature == "backbone":
        output = backbone(input_ids_pos=input_ids, input_ids_rc=input_ids_rc,
                          return_bottleneck=True)
        return output["bottleneck"], output
    if feature == "decoder":
        output = backbone(input_ids_pos=input_ids, input_ids_rc=input_ids_rc,
                          return_decoder_feature=True)
        decoder_feature = output.get("decoder_feature")
        if decoder_feature is None:
            raise ValueError("selected backbone did not return decoder_feature")
        return decoder_feature, output
    if feature == "base":
        output = backbone(input_ids_pos=input_ids, input_ids_rc=input_ids_rc,
                          return_base_feature=True)
        base_feature = output.get("base_feature")
        if base_feature is None:
            raise ValueError("selected backbone did not return base_feature")
        return base_feature, output
    raise ValueError(f"unknown feature '{feature}'; expected backbone, decoder, or base")


def _load_class_from_file(model_spec: str):
    path_text, class_name = str(model_spec).strip(), None
    candidate = Path(path_text)
    if ":" in path_text:
        possible_path, possible_class = path_text.rsplit(":", 1)
        if Path(possible_path).expanduser().is_file():
            path_text, class_name = possible_path, possible_class
            candidate = Path(path_text)
    candidate = candidate.expanduser().resolve()
    if not candidate.is_file() or candidate.suffix != ".py":
        raise FileNotFoundError(f"model file does not exist: {candidate}")

    # Files inside this project are imported as package modules so their
    # relative imports (for example in vericomp/modeling.py) keep working.
    project_root = Path(__file__).resolve().parents[1]
    try:
        relative = candidate.relative_to(project_root).with_suffix("")
    except ValueError:
        relative = None
    if relative is not None:
        module = import_module(".".join(relative.parts))
    else:
        module_name = "vericomp_dynamic_" + hashlib.sha1(str(candidate).encode()).hexdigest()[:12]
        spec = importlib.util.spec_from_file_location(module_name, candidate)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load model file: {candidate}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    if class_name:
        target = getattr(module, class_name, None)
        if not inspect.isclass(target) or not issubclass(target, nn.Module):
            raise TypeError(f"{class_name} in {candidate} is not a torch.nn.Module")
        return target

    preferred = {
        "modeling.py": "VeriCompPretrainModel",
        "causal_modeling.py": "CausalVeriCompNextTokenModel",
    }.get(candidate.name)
    if preferred and hasattr(module, preferred):
        return getattr(module, preferred)
    candidates = [
        target for name, target in vars(module).items()
        if inspect.isclass(target) and issubclass(target, nn.Module)
        and target.__module__ == module.__name__
    ]
    if len(candidates) != 1:
        names = ", ".join(target.__name__ for target in candidates)
        raise ValueError(
            f"model file {candidate} contains {len(candidates)} nn.Module classes ({names}); "
            "use /path/model.py:ClassName"
        )
    return candidates[0]


def build_backbone(model_name: str, config: VeriCompConfig):
    name = str(model_name).strip()
    model_path = Path(name.split(":", 1)[0]).expanduser()
    if model_path.suffix == ".py" or model_path.is_file():
        return _load_class_from_file(name)(config)
    aliases = {
        "vericomp": VeriCompPretrainModel,
        "mlm": VeriCompPretrainModel,
        "vericomp_mlm": VeriCompPretrainModel,
        "vericomp_causal": CausalVeriCompNextTokenModel,
        "causal": CausalVeriCompNextTokenModel,
    }
    if name in aliases:
        if name == "causal_local_skip":
            return CausalVeriCompNextTokenModel(config, local_skip=True)
        return aliases[name](config)
    if name == "causal_local_skip":
        return CausalVeriCompNextTokenModel(config, local_skip=True)
    if ":" not in name:
        available = ", ".join(sorted((*aliases, "causal_local_skip")))
        raise ValueError(f"unknown model '{name}'; use a .py file, one of {available}, or module:ClassName")
    module_name, class_name = name.split(":", 1)
    target = getattr(import_module(module_name), class_name)
    return target(config)
