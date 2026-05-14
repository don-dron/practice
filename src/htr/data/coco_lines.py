from __future__ import annotations

import hashlib
import json
import operator
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from htr.transforms import TrainAugmentation
from htr.cuda_line_batch import (
    LINE_PREP_JPEG_CUDA,
    LINE_PREP_KEY,
    LINE_PREP_PNG_CROP,
    LINE_PREP_TENSOR,
    LINE_PREP_UINT8,
    U8_KIND_COCO_CROP,
    U8_KIND_KEY,
)
from htr.line_width_limits import clamp_line_width_px

_LINES_PT_CACHE_MAGIC = "htr_lines_pt_v2"
_LINES_U8_CACHE_MAGIC = "htr_lines_pt_u8_v1"


def _file_bytes_look_like_png(data: bytes) -> bool:
    """Файл с расширением .jpg иногда на самом деле PNG (ошибка экспорта)."""
    return len(data) >= 4 and data[0] == 0x89 and data[1:4] == b"PNG"


def _lines_tensor_cache_key(
    fname: str,
    bbox: Tuple[float, float, float, float],
    img_height: int,
    max_width: Optional[int],
    min_crop_width: int,
    cache_namespace: str = "",
    *,
    u8_deferred: bool = False,
) -> str:
    fn = fname.replace("\\", "/")
    if cache_namespace:
        base = (cache_namespace, fn, bbox, img_height, max_width, min_crop_width)
    else:
        base = (fn, bbox, img_height, max_width, min_crop_width)
    tup = base + ("u8pre_cuda",) if u8_deferred else base
    payload = repr(tup).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _lines_tensor_cache_path(cache_root: Path, key_hex: str) -> Path:
    # подкаталог по префиксу — меньше нагрузка на FS при десятках тысяч файлов
    return cache_root / key_hex[:2] / f"{key_hex}.pt"


def _load_lines_tensor_cache(path: Path) -> Optional[Tuple[str, torch.Tensor, Union[int, Tuple[int, int]]]]:
    """(mode, tensor, meta): meta — seq_w для float или (crop_h,crop_w) для u8."""
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    mag = payload.get("magic")
    if mag == _LINES_PT_CACHE_MAGIC:
        im = payload.get("image")
        w_raw = payload.get("width")
        if not isinstance(im, torch.Tensor):
            return None
        try:
            w = operator.index(w_raw)
        except (TypeError, ValueError):
            return None
        return "float", im, int(w)
    if mag == _LINES_U8_CACHE_MAGIC:
        im = payload.get("image_u8")
        ch = payload.get("crop_h")
        cw = payload.get("crop_w")
        if not isinstance(im, torch.Tensor) or im.dtype != torch.uint8:
            return None
        try:
            hi = operator.index(ch)
            wi = operator.index(cw)
        except (TypeError, ValueError):
            return None
        return "u8", im, (int(hi), int(wi))
    return None


def disk_lines_try_u8_cache(
    cache_root: Optional[Path], key_hex: str
) -> Optional[Tuple[torch.Tensor, int, int]]:
    """Есть готовый u8-.pt на диске — вернуть (tensor, crop_h, crop_w); иначе None (пересчёт не нужен)."""
    if cache_root is None or not key_hex:
        return None
    cpath = _lines_tensor_cache_path(cache_root, key_hex)
    if not cpath.is_file():
        return None
    disk_hit = _load_lines_tensor_cache(cpath)
    if disk_hit is None:
        return None
    mode, payload, aux = disk_hit
    if mode != "u8":
        return None
    tup = aux if isinstance(aux, tuple) else (0, 0)
    ch_, cw_ = int(tup[0]), int(tup[1])
    if not isinstance(payload, torch.Tensor):
        return None
    if int(payload.shape[-2]) != ch_ or int(payload.shape[-1]) != cw_:
        return None
    return payload, ch_, cw_


def disk_lines_try_float_cache(
    cache_root: Optional[Path], key_hex: str
) -> Optional[Tuple[torch.Tensor, int]]:
    """Есть готовый float-.pt на диске — (tensor, width); иначе None."""
    if cache_root is None or not key_hex:
        return None
    cpath = _lines_tensor_cache_path(cache_root, key_hex)
    if not cpath.is_file():
        return None
    disk_hit = _load_lines_tensor_cache(cpath)
    if disk_hit is None:
        return None
    mode, payload, aux = disk_hit
    if mode != "float":
        return None
    wf = int(aux) if isinstance(aux, int) else 0
    if not isinstance(payload, torch.Tensor):
        return None
    return payload, wf


def _save_lines_tensor_cache(path: Path, image: torch.Tensor, width_px: int) -> None:
    # Не перезаписываем уже валидный .pt (другой воркер мог дописать кэш, пока мы декодировали).
    if path.is_file():
        hit = _load_lines_tensor_cache(path)
        if hit is not None and hit[0] == "float":
            _, payload, wf_raw = hit
            wf = int(wf_raw) if isinstance(wf_raw, int) else 0
            if isinstance(payload, torch.Tensor) and wf == int(width_px):
                return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save({"magic": _LINES_PT_CACHE_MAGIC, "image": image.detach().cpu().contiguous(), "width": width_px}, tmp)
    os.replace(tmp, path)


def _save_lines_u8_cache(path: Path, image_u8: torch.Tensor, crop_h: int, crop_w: int) -> None:
    if path.is_file():
        hit = _load_lines_tensor_cache(path)
        if hit is not None and hit[0] == "u8":
            _, payload, aux = hit
            tup = aux if isinstance(aux, tuple) else (0, 0)
            ch_, cw_ = int(tup[0]), int(tup[1])
            if (
                isinstance(payload, torch.Tensor)
                and ch_ == int(crop_h)
                and cw_ == int(crop_w)
                and int(payload.shape[-2]) == ch_
                and int(payload.shape[-1]) == cw_
            ):
                return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "magic": _LINES_U8_CACHE_MAGIC,
            "image_u8": image_u8.detach().cpu().contiguous(),
            "crop_h": int(crop_h),
            "crop_w": int(crop_w),
        },
        tmp,
    )
    os.replace(tmp, path)


class _RamTensorLRU:
    """LRU: float-после ресайза (совместимо с page_txt) или uint8-crop перед CUDA-ресайзом."""

    OVERHEAD_BYTES = 256

    def __init__(self, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes должен быть > 0")
        self._max_bytes = max_bytes
        self._cur = 0
        self._od: OrderedDict[str, dict] = OrderedDict()

    @staticmethod
    def payload_bytes(tensor: torch.Tensor) -> int:
        return int(tensor.numel() * tensor.element_size() + _RamTensorLRU.OVERHEAD_BYTES)

    def get_float(self, key: str) -> Optional[Tuple[torch.Tensor, int]]:
        """Как раньше: (tensor, width)."""
        rec = self._od.get(key)
        if rec is None or rec.get("kind") != "float":
            return None
        self._od.move_to_end(key)
        return rec["tensor"], int(rec["width"])

    def put_float(self, key: str, tensor: torch.Tensor, width_px: int) -> None:
        t = tensor.detach().cpu().clone().contiguous()
        self._put_any(key, "float", t, width=int(width_px))

    def get_u8(self, key: str) -> Optional[Tuple[torch.Tensor, int, int]]:
        rec = self._od.get(key)
        if rec is None or rec.get("kind") != "u8":
            return None
        self._od.move_to_end(key)
        return rec["tensor"], int(rec["ch"]), int(rec["cw"])

    def put_u8(self, key: str, u8: torch.Tensor, ch: int, cw: int) -> None:
        t = u8.detach().cpu().clone().contiguous()
        self._put_any(key, "u8", t, ch=int(ch), cw=int(cw))

    def _put_any(self, key: str, kind: str, tensor: torch.Tensor, **extra: object) -> None:
        need = self.payload_bytes(tensor)
        if need > self._max_bytes:
            return

        prev = self._od.pop(key, None)
        if prev is not None:
            self._cur -= self.payload_bytes(prev["tensor"])

        while self._cur + need > self._max_bytes and self._od:
            _, prev2 = self._od.popitem(last=False)
            self._cur -= self.payload_bytes(prev2["tensor"])

        if self._cur + need <= self._max_bytes:
            rec: dict = {"kind": kind, "tensor": tensor, **extra}
            self._od[key] = rec
            self._cur += need
            self._od.move_to_end(key)


def _bbox_clip(x: float, y: float, w: float, h: float, W: int, H: int) -> Tuple[float, float, float, float]:
    x2, y2 = x + w, y + h
    x = max(0.0, min(x, float(W)))
    y = max(0.0, min(y, float(H)))
    x2 = max(0.0, min(x2, float(W)))
    y2 = max(0.0, min(y2, float(H)))
    return x, y, max(1.0, x2 - x), max(1.0, y2 - y)


class COCOLinesDataset(Dataset):
    """
    Строковые наблюдения из COCO: crop по bbox, текст из attributes[text_field].
    Поле samples публично — для сборки алфавита после сплита.
    """

    def __init__(
        self,
        coco_json: Union[str, Path],
        image_root: Union[str, Path],
        text_field: str = "translation",
        img_height: int = 32,
        max_width: Optional[int] = 1200,
        min_crop_width: int = 4,
        train_augmentation: Optional[TrainAugmentation] = None,
        preprocessed_cache_dir: Optional[Union[str, Path]] = None,
        preprocessed_ram_cache_max_bytes: Optional[int] = None,
        cache_namespace: str = "",
        defer_resize_normalize_to_cuda: bool = False,
        jpeg_decode_cuda_workers_zero: bool = False,
    ):
        self.image_root = Path(image_root)
        self.text_field = text_field
        self.img_height = img_height
        self.max_width = max_width
        self.min_crop_width = min_crop_width
        self.train_augment = train_augmentation
        self._defer_resize_cuda = bool(defer_resize_normalize_to_cuda)
        self._jpeg_cuda = bool(jpeg_decode_cuda_workers_zero)
        self._cache_namespace = (cache_namespace or "").strip()
        rc = preprocessed_cache_dir
        self.preprocessed_cache_root: Optional[Path] = None
        if rc is not None:
            s = str(rc).strip()
            if s:
                self.preprocessed_cache_root = Path(s).expanduser().resolve()

        _ram_budget = preprocessed_ram_cache_max_bytes
        self._ram_lru: Optional[_RamTensorLRU] = None
        if _ram_budget is not None and int(_ram_budget) > 0:
            self._ram_lru = _RamTensorLRU(int(_ram_budget))

        coco_json_path = Path(coco_json)
        with open(coco_json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        id_to_file = {im["id"]: im["file_name"] for im in coco["images"]}
        samples: List[Tuple[str, str, Tuple[float, float, float, float]]] = []
        for ann in coco["annotations"]:
            image_id = ann["image_id"]
            if image_id not in id_to_file:
                continue
            bbox = ann.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue
            attrs = ann.get("attributes") or {}
            txt = attrs.get(text_field)
            if txt is None or not str(txt).strip():
                continue
            fname = id_to_file[image_id]
            samples.append((fname, str(txt).strip(), tuple(float(t) for t in bbox)))

        self.samples = samples

        if self._jpeg_cuda and self.train_augment is not None:
            raise ValueError("jpeg_decode_cuda_workers_zero несовместим с train_augmentation.")
        if self._defer_resize_cuda and self.train_augment is not None:
            raise ValueError(
                "defer_resize_normalize_to_cuda несовместим с train_augmentation "
                "(аугментации должны остаться на CPU с PIL)."
            )
        if self.preprocessed_cache_root is not None and self.train_augment is not None:
            raise ValueError(
                "preprocessed_cache_dir задан совместно с train_augmentation: кэш был бы "
                "недетерминированным при аугментациях — отключите один из них."
            )
        if self._ram_lru is not None and self.train_augment is not None:
            raise ValueError(
                "RAM-кэш строк (preprocessed_ram_cache_max_gb / max_bytes) несовместим с augmentation_train."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _prepare_tensor(self, crop: Image.Image) -> Tuple[torch.Tensor, int]:
        if self.train_augment is not None:
            crop = self.train_augment(crop)
        else:
            crop = TF.rgb_to_grayscale(crop, num_output_channels=1)

        w_t, h_t = crop.size
        new_w = clamp_line_width_px(w_t * self.img_height / h_t, max_width=self.max_width)
        crop_resized = crop.resize((new_w, self.img_height), Image.Resampling.BILINEAR)
        tens = TF.to_tensor(crop_resized)
        tens = TF.normalize(tens, mean=[0.5], std=[0.5])
        out_w = tens.shape[-1]
        return tens, out_w

    def _decode_crop_resize_torchvision(self, img_path: Path, bbox: Tuple[float, float, float, float]) -> Optional[Tuple[torch.Tensor, int]]:
        """JPEG/PNG через torch (decodeJPEG), без PIL если нет аугментаций — ниже простой нагрузке CPU."""
        try:
            from torchvision.io import read_image
        except Exception:
            return None
        try:
            chw = read_image(str(img_path))
        except Exception:
            return None
        if chw.dtype != torch.uint8 or chw.dim() != 3:
            return None
        ch = int(chw.shape[0])
        if ch == 4:
            chw = chw[:3]
        elif ch not in (1, 3):
            return None
        H, W = int(chw.shape[1]), int(chw.shape[2])
        x, y, bw, bh = _bbox_clip(*bbox, W, H)
        if bw < float(self.min_crop_width):
            bw = float(self.min_crop_width)
            if x + bw > W:
                x = float(max(0, W - self.min_crop_width))
        left = int(x)
        top = int(y)
        right = int(x + bw)
        bottom = int(y + bh)
        left = max(0, min(left, max(0, W - 1)))
        top = max(0, min(top, max(0, H - 1)))
        right = max(left + 1, min(W, right))
        bottom = max(top + 1, min(H, bottom))
        sl = chw[:, top:bottom, left:right]
        if sl.numel() == 0:
            return None
        x01 = sl.float().div_(255.0)
        if sl.shape[0] == 3:
            gray = 0.2989 * x01[0:1] + 0.5870 * x01[1:2] + 0.1140 * x01[2:3]
        else:
            gray = x01
        _, hi, wi_src = gray.shape
        new_w = clamp_line_width_px(
            float(wi_src) * float(self.img_height) / float(hi), max_width=self.max_width
        )
        out = F.interpolate(
            gray.unsqueeze(0),
            size=(self.img_height, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        out = out.mul_(2.0).sub_(1.0)
        return out, new_w

    @staticmethod
    def _sl_rgb_to_gray_u8(sl: torch.Tensor) -> torch.Tensor:
        x01 = sl.float().div_(255.0)
        if sl.shape[0] == 3:
            gray01 = 0.2989 * x01[0:1] + 0.5870 * x01[1:2] + 0.1140 * x01[2:3]
        else:
            gray01 = x01
        return (gray01 * 255.0).clamp(0.0, 255.0).to(torch.uint8)

    def _decode_crop_gray_u8_torchvision(self, img_path: Path, bbox: Tuple[float, float, float, float]) -> Optional[torch.Tensor]:
        try:
            from torchvision.io import read_image
        except Exception:
            return None
        try:
            chw = read_image(str(img_path))
        except Exception:
            return None
        if chw.dtype != torch.uint8 or chw.dim() != 3:
            return None
        ch = int(chw.shape[0])
        if ch == 4:
            chw = chw[:3]
        elif ch not in (1, 3):
            return None
        H, W = int(chw.shape[1]), int(chw.shape[2])
        x, y, bw, bh = _bbox_clip(*bbox, W, H)
        if bw < float(self.min_crop_width):
            bw = float(self.min_crop_width)
            if x + bw > W:
                x = float(max(0, W - self.min_crop_width))
        left = int(x)
        top = int(y)
        right = int(x + bw)
        bottom = int(y + bh)
        left = max(0, min(left, max(0, W - 1)))
        top = max(0, min(top, max(0, H - 1)))
        right = max(left + 1, min(W, right))
        bottom = max(top + 1, min(H, bottom))
        sl = chw[:, top:bottom, left:right]
        if sl.numel() == 0:
            return None
        return self._sl_rgb_to_gray_u8(sl)

    def _pil_crop_gray_u8(self, crop: Image.Image) -> torch.Tensor:
        pil_g = TF.rgb_to_grayscale(crop.convert("RGB"), num_output_channels=1)
        arr = np.asarray(pil_g, dtype=np.uint8)
        return torch.from_numpy(arr).unsqueeze(0).contiguous()

    def _load_sample_gray_u8(self, img_path: Path, bbox: Tuple[float, float, float, float]) -> torch.Tensor:
        if self.train_augment is None:
            tv = self._decode_crop_gray_u8_torchvision(img_path, bbox)
            if tv is not None:
                return tv
        pil = Image.open(img_path).convert("RGB")
        W, H = pil.size
        x, y, w, h = _bbox_clip(*bbox, W, H)
        if w < float(self.min_crop_width):
            w = float(self.min_crop_width)
            if x + w > W:
                x = float(max(0, W - self.min_crop_width))
        region = pil.crop((int(x), int(y), int(x + w), int(y + h)))
        return self._pil_crop_gray_u8(region)

    def _load_sample_tensors(self, img_path: Path, bbox: Tuple[float, float, float, float]) -> Tuple[torch.Tensor, int]:
        if self.train_augment is None:
            tv = self._decode_crop_resize_torchvision(img_path, bbox)
            if tv is not None:
                return tv
        pil = Image.open(img_path).convert("RGB")
        W, H = pil.size
        x, y, w, h = _bbox_clip(*bbox, W, H)
        if w < float(self.min_crop_width):
            w = float(self.min_crop_width)
            if x + w > W:
                x = float(max(0, W - self.min_crop_width))
        crop = pil.crop((int(x), int(y), int(x + w), int(y + h)))
        return self._prepare_tensor(crop)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        fname, text, bbox = self.samples[idx]
        img_path = self.image_root / fname

        ram = self._ram_lru
        cache_root = self.preprocessed_cache_root
        defer = self._defer_resize_cuda
        use_key = ram is not None or cache_root is not None

        key_hex = (
            _lines_tensor_cache_key(
                fname,
                bbox,
                self.img_height,
                self.max_width,
                self.min_crop_width,
                self._cache_namespace,
                u8_deferred=defer,
            )
            if use_key
            else ""
        )

        def _tensor_item_cpu(tens: torch.Tensor, out_w: int) -> Dict[str, Union[torch.Tensor, str]]:
            return {
                LINE_PREP_KEY: LINE_PREP_TENSOR,
                "image": tens.contiguous(),
                "width": torch.tensor(out_w, dtype=torch.long),
                "text": text,
            }

        def _u8_item(u8_crop: torch.Tensor) -> Dict[str, Union[torch.Tensor, str, int]]:
            hi = int(u8_crop.shape[-2])
            wi = int(u8_crop.shape[-1])
            return {
                LINE_PREP_KEY: LINE_PREP_UINT8,
                "image_u8": u8_crop.contiguous(),
                "crop_h": hi,
                "crop_w": wi,
                U8_KIND_KEY: U8_KIND_COCO_CROP,
                "text": text,
            }

        if defer:
            if ram is not None and key_hex:
                hit_u8 = ram.get_u8(key_hex)
                if hit_u8 is not None:
                    tens_u8, ch, cw = hit_u8
                    if int(tens_u8.shape[-2]) == ch and int(tens_u8.shape[-1]) == cw:
                        return _u8_item(tens_u8)

            def _take_u8_from_disk() -> Optional[Dict[str, Union[torch.Tensor, str, int]]]:
                u8t = disk_lines_try_u8_cache(cache_root, key_hex)
                if u8t is None:
                    return None
                payload, ch_, cw_ = u8t
                if ram is not None:
                    ram.put_u8(key_hex, payload, ch_, cw_)
                return _u8_item(payload)

            got = _take_u8_from_disk()
            if got is not None:
                return got

            # Промах u8 по RAM/диску. Быстрый путь: сырая JPEG/PNG в collate (nvJPEG/decode_png на GPU)
            # — но тогда .pt на диск не пишется и каждая эпоха снова платит за чтение/декод.
            # Если включён preprocessed_cache_dir — один раз грузим u8 на CPU, сохраняем .pt, дальше только диск.
            if self._jpeg_cuda and self.train_augment is None:
                sfx = Path(fname).suffix.lower()
                disk_wants_pt = cache_root is not None and bool(key_hex)
                if not disk_wants_pt:
                    if sfx in (".jpg", ".jpeg"):
                        data = img_path.read_bytes()
                        jb = torch.frombuffer(bytearray(data), dtype=torch.uint8)
                        if _file_bytes_look_like_png(data):
                            return {
                                LINE_PREP_KEY: LINE_PREP_PNG_CROP,
                                "png_bytes": jb,
                                "bbox": bbox,
                                "text": text,
                            }
                        return {
                            LINE_PREP_KEY: LINE_PREP_JPEG_CUDA,
                            "jpeg_bytes": jb,
                            "bbox": bbox,
                            "text": text,
                        }
                    if sfx == ".png":
                        data = img_path.read_bytes()
                        jb = torch.frombuffer(bytearray(data), dtype=torch.uint8)
                        return {
                            LINE_PREP_KEY: LINE_PREP_PNG_CROP,
                            "png_bytes": jb,
                            "bbox": bbox,
                            "text": text,
                        }

            got2 = _take_u8_from_disk()
            if got2 is not None:
                return got2

            gray_u8 = self._load_sample_gray_u8(img_path, bbox)
            if ram is not None and key_hex:
                _1c, hh, ww = gray_u8.shape
                ram.put_u8(key_hex, gray_u8, int(hh), int(ww))
            if cache_root is not None and key_hex:
                cpath = _lines_tensor_cache_path(cache_root, key_hex)
                _1x, hh2, ww2 = gray_u8.shape
                try:
                    _save_lines_u8_cache(cpath, gray_u8, int(hh2), int(ww2))
                except Exception:
                    pass
            return _u8_item(gray_u8)

        if ram is not None and key_hex:
            hit_ram = ram.get_float(key_hex)
            if hit_ram is not None:
                tens_f, ow = hit_ram
                return _tensor_item_cpu(tens_f, ow)

        def _take_float_from_disk() -> Optional[Dict[str, Union[torch.Tensor, str]]]:
            ft = disk_lines_try_float_cache(cache_root, key_hex)
            if ft is None:
                return None
            payload, wf = ft
            if ram is not None:
                ram.put_float(key_hex, payload, wf)
            return _tensor_item_cpu(payload, wf)

        gotf = _take_float_from_disk()
        if gotf is not None:
            return gotf

        tens, out_w = self._load_sample_tensors(img_path, bbox)

        if ram is not None and key_hex:
            ram.put_float(key_hex, tens, out_w)

        if cache_root is not None and key_hex:
            cpath = _lines_tensor_cache_path(cache_root, key_hex)
            try:
                _save_lines_tensor_cache(cpath, tens, out_w)
            except Exception:
                pass

        return _tensor_item_cpu(tens, out_w)


def coco_collate_fn(batch: List[Dict[str, Union[torch.Tensor, str]]]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    texts = [b["text"] for b in batch]  # type: ignore[list-item]
    widths_list = [int(b["width"].item()) for b in batch]  # type: ignore[arg-type]
    target_h = batch[0]["image"].shape[-2]
    pad_w = max(widths_list)
    images = torch.zeros(len(batch), 1, target_h, pad_w, dtype=batch[0]["image"].dtype)  # type: ignore[arg-type,index]
    for i, b in enumerate(batch):
        im = b["image"]  # type: ignore[misc,index]
        w = im.shape[-1]
        images[i, :, :, :w] = im
    return {"image": images, "text": texts, "seq_width": torch.tensor(widths_list, dtype=torch.long)}
