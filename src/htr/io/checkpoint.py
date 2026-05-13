from __future__ import annotations

import torch


def save_checkpoint(
    path: str,
    state_dict: dict,
    *,
    itos: list[str],
    model_name: str,
    yaml_dump: dict,
) -> None:
    torch.save({"state_dict": state_dict, "itos": itos, "model_name": model_name, "config_yaml": yaml_dump}, path)


def load_checkpoint(path: str) -> dict:
    kw: dict = {}
    try:
        import inspect

        if "weights_only" in inspect.signature(torch.load).parameters:
            kw["weights_only"] = False
    except (TypeError, ValueError):
        pass
    return torch.load(path, map_location="cpu", **kw)
