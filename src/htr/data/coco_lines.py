from __future__ import annotations

import hashlib
import json
import operator
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from htr.transforms import TrainAugmentation

_LINES_PT_CACHE_MAGIC = "htr_lines_pt_v1"


def _lines_tensor_cache_key(
    fname: str,
    bbox: Tuple[float, float, float, float],
    img_height: int,
    max_width: Optional[int],
    min_crop_width: int,
) -> str:
    fn = fname.replace("\\", "/")
    payload = repr((fn, bbox, img_height, max_width, min_crop_width)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _lines_tensor_cache_path(cache_root: Path, key_hex: str) -> Path:
    # подкаталог по префиксу — меньше нагрузка на FS при десятках тысяч файлов
    return cache_root / key_hex[:2] / f"{key_hex}.pt"


def _load_lines_tensor_cache(path: Path) -> Optional[Tuple[torch.Tensor, int]]:
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("magic") != _LINES_PT_CACHE_MAGIC:
        return None
    im = payload.get("image")
    w_raw = payload.get("width")
    if not isinstance(im, torch.Tensor):
        return None
    try:
        w = operator.index(w_raw)
    except (TypeError, ValueError):
        return None
    return im, int(w)


def _save_lines_tensor_cache(path: Path, image: torch.Tensor, width_px: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save({"magic": _LINES_PT_CACHE_MAGIC, "image": image.detach().cpu().contiguous(), "width": width_px}, tmp)
    os.replace(tmp, path)


class _RamTensorLRU:
    """LRU хранилище тензоров строк с лимитом по приблизительному числу байт."""

    OVERHEAD_BYTES = 256

    def __init__(self, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes должен быть > 0")
        self._max_bytes = max_bytes
        self._cur = 0
        self._od: OrderedDict[str, Tuple[torch.Tensor, int]] = OrderedDict()

    @staticmethod
    def payload_bytes(tensor: torch.Tensor) -> int:
        return int(tensor.numel() * tensor.element_size() + _RamTensorLRU.OVERHEAD_BYTES)

    def get(self, key: str) -> Optional[Tuple[torch.Tensor, int]]:
        pair = self._od.get(key)
        if pair is None:
            return None
        self._od.move_to_end(key)
        return pair

    def put(self, key: str, tensor: torch.Tensor, width_px: int) -> None:
        t = tensor.detach().cpu().clone().contiguous()
        need = self.payload_bytes(t)
        if need > self._max_bytes:
            return

        prev = self._od.pop(key, None)
        if prev is not None:
            self._cur -= self.payload_bytes(prev[0])

        while self._cur + need > self._max_bytes and self._od:
            _, (ev_t, _) = self._od.popitem(last=False)
            self._cur -= self.payload_bytes(ev_t)

        if self._cur + need <= self._max_bytes:
            self._od[key] = (t, int(width_px))
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
    ):
        self.image_root = Path(image_root)
        self.text_field = text_field
        self.img_height = img_height
        self.max_width = max_width
        self.min_crop_width = min_crop_width
        self.train_augment = train_augmentation
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
        new_w = max(1, round(w_t * self.img_height / h_t))
        if self.max_width is not None and new_w > self.max_width:
            new_w = self.max_width
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
        new_w = max(1, round(float(wi_src) * float(self.img_height) / float(hi)))
        if self.max_width is not None:
            new_w = min(new_w, int(self.max_width))
        out = F.interpolate(
            gray.unsqueeze(0),
            size=(self.img_height, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        out = out.mul_(2.0).sub_(1.0)
        return out, new_w

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
        use_key = ram is not None or cache_root is not None

        key_hex = (
            _lines_tensor_cache_key(fname, bbox, self.img_height, self.max_width, self.min_crop_width) if use_key else ""
        )

        def _tensor_item(tens: torch.Tensor, out_w: int) -> Dict[str, Union[torch.Tensor, str]]:
            return {
                "image": tens.contiguous(),
                "width": torch.tensor(out_w, dtype=torch.long),
                "text": text,
            }

        if ram is not None:
            hit = ram.get(key_hex)
            if hit is not None:
                tens, out_w = hit
                return _tensor_item(tens, out_w)

        if cache_root is not None:
            cpath = _lines_tensor_cache_path(cache_root, key_hex)
            disk_hit = _load_lines_tensor_cache(cpath)
            if disk_hit is not None:
                tens, out_w = disk_hit
                if ram is not None:
                    ram.put(key_hex, tens, out_w)
                return _tensor_item(tens, out_w)

        tens, out_w = self._load_sample_tensors(img_path, bbox)

        if ram is not None and key_hex:
            ram.put(key_hex, tens, out_w)

        if cache_root is not None and key_hex:
            cpath = _lines_tensor_cache_path(cache_root, key_hex)
            try:
                _save_lines_tensor_cache(cpath, tens, out_w)
            except Exception:
                pass

        return _tensor_item(tens, out_w)


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
