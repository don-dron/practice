from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, List, Optional, Union

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: Union[Path, str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(paths: List[Union[Path, str]]) -> dict[str, Any]:
    if not paths:
        raise ValueError("нужен хотя бы один YAML-конфиг")
    merged: dict[str, Any] = {}
    for p in paths:
        merged = deep_merge(merged, load_yaml(p))
    return merged


def config_paths_from_args(default_name: str, config_arg: Optional[List[str]] = None) -> List[Path]:
    root = Path(__file__).resolve().parents[2]
    default_main = root / "configs" / default_name
    if not config_arg:
        return [default_main]
    out: list[Path] = []
    for raw in config_arg:
        path = Path(raw)
        if not path.is_absolute():
            cand = root / path
            path = cand if cand.exists() else path
        out.append(path)
    return out
