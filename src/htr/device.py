from __future__ import annotations

from typing import Optional

import torch


def pick_device(name: str) -> str:
    name = name.lower().strip()
    if name == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "mps":
        ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        return "mps" if ok else "cpu"
    return "cpu"


def move_training_image_batch(
    batch: dict,
    device: torch.device,
    *,
    img_height: int,
    max_width: Optional[int],
) -> dict:
    """После collate: ресайз линии + нормализация на CUDA при uint8-пакете (см. cuda_line_batch)."""
    from htr.cuda_line_batch import (
        HTR_BATCH_ON_DEVICE,
        finalize_line_batch_cuda,
        line_batch_needs_cuda_finalize,
    )

    batch_work = {k: v for k, v in batch.items() if k != HTR_BATCH_ON_DEVICE}
    prepped = batch.get(HTR_BATCH_ON_DEVICE)
    if prepped is True:
        return batch_work

    if line_batch_needs_cuda_finalize(batch_work):
        return finalize_line_batch_cuda(batch_work, device=device, img_height=img_height, max_width=max_width)
    return move_batch_to_device(batch_work, device)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = dict(batch)
    img = batch["image"]
    if isinstance(img, torch.Tensor):
        out["image"] = img.to(device, non_blocking=True)
    sq = batch.get("seq_width")
    if isinstance(sq, torch.Tensor):
        out["seq_width"] = sq.to(device, non_blocking=True)
    return out
