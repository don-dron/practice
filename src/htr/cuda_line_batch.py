"""Перенос интерполяции и нормализации линии на CUDA (JPEG-декод по-прежнему на CPU в воркерах даталоадера)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union, cast

import torch
import torch.nn.functional as F

LINE_PREP_KEY = "line_prep"
LINE_PREP_TENSOR = "tensor"
LINE_PREP_UINT8 = "uint8"
LINE_PREP_JPEG_CUDA = "jpeg_cuda"
U8_KIND_KEY = "u8_kind"
U8_KIND_COCO_CROP = "coco_crop"
U8_KIND_PAGE_READY = "page_ready"
HTR_BATCH_ON_DEVICE = "_htr_lines_on_device"


def line_batch_needs_cuda_finalize(batch: dict) -> bool:
    return batch.get(LINE_PREP_KEY) == LINE_PREP_UINT8


def finalize_line_batch_cuda(
    batch: dict,
    *,
    device: torch.device,
    img_height: int,
    max_width: Optional[int],
) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """uint8-пакет из collate → float32 [-1,1] на device, seq_width на device."""
    padded = batch["image_u8"]
    h = batch["crop_h"].to(device, non_blocking=True)
    w = batch["crop_w"].to(device, non_blocking=True)
    kind_raw = batch[U8_KIND_KEY]
    texts: List[str] = batch["text"]  # type: ignore[assignment]

    bsz = int(padded.shape[0])
    if isinstance(kind_raw, (list, tuple)):
        kinds: List[str] = [str(x) for x in cast(Sequence[object], kind_raw)]
        if len(kinds) != bsz:
            raise ValueError(f"u8_kind: ожидалось {bsz} меток, получено {len(kinds)}")
    else:
        k0 = str(kind_raw)
        if k0 not in (U8_KIND_COCO_CROP, U8_KIND_PAGE_READY):
            raise ValueError(f"неизвестный u8_kind={k0!r}")
        kinds = [k0] * bsz

    for ki in kinds:
        if ki not in (U8_KIND_COCO_CROP, U8_KIND_PAGE_READY):
            raise ValueError(f"неизвестный u8_kind={ki!r}")

    out_slices: List[torch.Tensor] = []
    new_ws: List[int] = []
    for i in range(bsz):
        hi = int(h[i].item())
        wi = int(w[i].item())
        sl = padded[i : i + 1, :, :hi, :wi].to(device, non_blocking=True).float().div_(255.0)
        if kinds[i] == U8_KIND_COCO_CROP:
            new_w = max(1, round(float(wi) * float(img_height) / float(max(hi, 1))))
            if max_width is not None:
                new_w = min(new_w, int(max_width))
            y = F.interpolate(sl, size=(img_height, new_w), mode="bilinear", align_corners=False)
        else:
            # page_ready: уже высота img_height
            y = sl
        y = y.mul_(2.0).sub_(1.0)
        out_slices.append(y)
        new_ws.append(int(y.shape[-1]))

    pad_w = max(new_ws)
    target_h = img_height
    images = torch.zeros(bsz, 1, target_h, pad_w, dtype=out_slices[0].dtype, device=device)
    seq_width = torch.zeros(bsz, dtype=torch.long, device=device)
    for i, yi in enumerate(out_slices):
        w_i = yi.shape[-1]
        images[i, :, :, :w_i] = yi.squeeze(0)
        seq_width[i] = int(w_i)

    return {"image": images, "text": texts, "seq_width": seq_width}


def coco_collate_mixed_lines(batch: List[dict]) -> Dict[str, Union[torch.Tensor, List[str], str]]:
    """Поддерживает элементы coco (u8 bbox-crop) и page_txt (u8 уже по высоте).

    ConcatDataset + shuffle смешивают источники в одном батче — для каждой строки свой u8_kind (или одна строка, если все одного типа).
    """
    from htr.data.coco_lines import coco_collate_fn

    modes = [str(b.get(LINE_PREP_KEY, LINE_PREP_TENSOR)) for b in batch]
    if any(m != modes[0] for m in modes):
        raise RuntimeError(
            "в одном батче смешаны line_prep режимы (tensor vs uint8) — конфликт ConcatDataset или баг данных"
        )
    if modes[0] == LINE_PREP_TENSOR:
        return coco_collate_fn(batch)

    kinds = [str(b[U8_KIND_KEY]) for b in batch]
    for k in kinds:
        if k not in (U8_KIND_COCO_CROP, U8_KIND_PAGE_READY):
            raise ValueError(f"неизвестный u8_kind={k!r}")

    texts = [b["text"] for b in batch]  # type: ignore[list-item]
    u8_tensors = [b["image_u8"] for b in batch]  # CHW uint8
    hs = torch.tensor([int(b["crop_h"]) for b in batch], dtype=torch.long)
    ws = torch.tensor([int(b["crop_w"]) for b in batch], dtype=torch.long)

    mh = max(t.shape[-2] for t in u8_tensors)
    mw = max(t.shape[-1] for t in u8_tensors)
    out = torch.zeros(len(batch), 1, mh, mw, dtype=torch.uint8)
    for i, t in enumerate(u8_tensors):
        _, th, tw = t.shape
        out[i, :, :th, :tw] = t

    kind_field: Union[str, List[str]] = kinds[0] if all(x == kinds[0] for x in kinds) else kinds

    return {
        LINE_PREP_KEY: LINE_PREP_UINT8,
        "image_u8": out,
        "crop_h": hs,
        "crop_w": ws,
        "u8_kind": kind_field,
        "text": texts,
    }


def _bbox_clip_lines(x: float, y: float, w: float, h: float, W: int, H: int) -> Tuple[float, float, float, float]:
    x2, y2 = x + w, y + h
    x = max(0.0, min(x, float(W)))
    y = max(0.0, min(y, float(H)))
    x2 = max(0.0, min(x2, float(W)))
    y2 = max(0.0, min(y2, float(H)))
    return x, y, max(1.0, x2 - x), max(1.0, y2 - y)


def _line_tensor_from_rgb_chw_cuda(
    chw_uint8: torch.Tensor,
    bbox: Tuple[float, float, float, float],
    *,
    img_height: int,
    max_width: Optional[int],
    min_crop_width: float,
) -> torch.Tensor:
    """Полная страница RGB uint8 на GPU → линия float [-1,1], форма [1,1,img_height,new_w]."""
    H, W = int(chw_uint8.shape[1]), int(chw_uint8.shape[2])
    bx, by, bw, bh = _bbox_clip_lines(*bbox, W, H)
    if bw < float(min_crop_width):
        bw = float(min_crop_width)
        if bx + bw > W:
            bx = float(max(0, W - min_crop_width))
    left = int(bx)
    top = int(by)
    right = int(bx + bw)
    bottom = int(by + bh)
    left = max(0, min(left, max(0, W - 1)))
    top = max(0, min(top, max(0, H - 1)))
    right = max(left + 1, min(W, right))
    bottom = max(top + 1, min(H, bottom))
    sl = chw_uint8[:, top:bottom, left:right]
    if sl.numel() == 0:
        raise RuntimeError("пустой crop после bbox")
    x01 = sl.float().div_(255.0)
    if sl.shape[0] == 3:
        gray01 = 0.2989 * x01[0:1] + 0.5870 * x01[1:2] + 0.1140 * x01[2:3]
    else:
        gray01 = x01
    _, hi, wi_src = gray01.shape
    new_w = max(1, round(float(wi_src) * float(img_height) / float(max(hi, 1))))
    if max_width is not None:
        new_w = min(new_w, int(max_width))
    y = F.interpolate(gray01.unsqueeze(0), size=(img_height, new_w), mode="bilinear", align_corners=False).squeeze(0)
    y = y.mul_(2.0).sub_(1.0)
    return y.unsqueeze(0)


def collate_gpu_lines_jpeg_cuda_batch(
    batch: List[dict],
    *,
    device: torch.device,
    img_height: int,
    max_width: Optional[int],
    min_crop_width: int,
) -> Dict[str, Union[torch.Tensor, List[str], bool, str]]:
    """JPEG: decode_jpeg(..., device=cuda) списком; PNG/u8: finalize на GPU. Итог на device."""
    from htr.data.coco_lines import coco_collate_fn

    modes = [str(b.get(LINE_PREP_KEY, LINE_PREP_TENSOR)) for b in batch]
    if all(m == LINE_PREP_TENSOR for m in modes):
        out: Dict[str, Union[torch.Tensor, List[str], bool]] = dict(coco_collate_fn(batch))
        out[HTR_BATCH_ON_DEVICE] = False
        return out

    if device.type != "cuda":
        raise RuntimeError("collate_gpu_lines_jpeg_cuda_batch рассчитан на CUDA decode_jpeg")

    bsz = len(batch)
    texts = [b["text"] for b in batch]  # type: ignore[list-item]

    idx_jpeg = [i for i, m in enumerate(modes) if m == LINE_PREP_JPEG_CUDA]
    idx_u8 = [i for i, m in enumerate(modes) if m == LINE_PREP_UINT8]

    for m in modes:
        if m not in (LINE_PREP_JPEG_CUDA, LINE_PREP_UINT8):
            raise RuntimeError(f"неожиданный line_prep в jpeg-collate: {m!r}")

    row_tensors: List[Optional[torch.Tensor]] = [None] * bsz

    if idx_jpeg:
        try:
            from torchvision.io import decode_jpeg
        except Exception as ex:
            raise RuntimeError("torchvision.io.decode_jpeg недоступен") from ex
        byte_tensors = [batch[i]["jpeg_bytes"] for i in idx_jpeg]  # type: ignore[misc]
        decoded = decode_jpeg(byte_tensors, device=device)
        if not isinstance(decoded, (list, tuple)):
            decoded = [decoded]
        if len(decoded) != len(idx_jpeg):
            raise RuntimeError("decode_jpeg: число кадров не совпало с числом JPEG в батче")
        for idx_flat, img_tensor in zip(idx_jpeg, decoded):
            row_tensors[idx_flat] = _line_tensor_from_rgb_chw_cuda(
                img_tensor,
                batch[idx_flat]["bbox"],  # type: ignore[index]
                img_height=img_height,
                max_width=max_width,
                min_crop_width=float(min_crop_width),
            )

    if idx_u8:
        sub = [batch[i] for i in idx_u8]
        u8_packed = coco_collate_mixed_lines(sub)
        fin = finalize_line_batch_cuda(u8_packed, device=device, img_height=img_height, max_width=max_width)
        imgs_u = fin["image"]
        sqw = fin["seq_width"]
        for j, global_i in enumerate(idx_u8):
            sw = int(sqw[j].item())
            row_tensors[global_i] = imgs_u[j : j + 1, :, :, :sw].contiguous()

    assert all(r is not None for r in row_tensors)
    results: List[torch.Tensor] = [row_tensors[i] for i in range(bsz)]  # type: ignore[misc]

    widths_list = [int(results[i].shape[-1]) for i in range(bsz)]
    pad_w = max(widths_list)
    images = torch.zeros(bsz, 1, img_height, pad_w, dtype=results[0].dtype, device=device)
    seq_width = torch.tensor(widths_list, dtype=torch.long, device=device)
    for i in range(bsz):
        wi = widths_list[i]
        images[i, :, :, :wi] = results[i].squeeze(0)

    return {
        LINE_PREP_KEY: LINE_PREP_TENSOR,
        "image": images,
        "text": texts,
        "seq_width": seq_width,
        HTR_BATCH_ON_DEVICE: True,
    }
