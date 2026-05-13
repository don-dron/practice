from __future__ import annotations

import torch


def pick_device(name: str) -> str:
    name = name.lower().strip()
    if name == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "mps":
        ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        return "mps" if ok else "cpu"
    return "cpu"


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = dict(batch)
    img = batch["image"]
    if isinstance(img, torch.Tensor):
        out["image"] = img.to(device, non_blocking=True)
    sq = batch.get("seq_width")
    if isinstance(sq, torch.Tensor):
        out["seq_width"] = sq.to(device, non_blocking=True)
    return out
