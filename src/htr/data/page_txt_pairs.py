"""Пары страница PNG + plaintext TXT (дореформенный OCR-подкаталог без COCO bbox). Формат выхода совместим с coco_collate_fn."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from htr.data.coco_lines import (
    _RamTensorLRU,
    _lines_tensor_cache_path,
    _save_lines_tensor_cache,
    _save_lines_u8_cache,
    disk_lines_try_float_cache,
    disk_lines_try_u8_cache,
)
from htr.cuda_line_batch import (
    LINE_PREP_KEY,
    LINE_PREP_PNG_PAGE,
    LINE_PREP_TENSOR,
    LINE_PREP_UINT8,
    U8_KIND_KEY,
    U8_KIND_PAGE_READY,
)
from htr.transforms import TrainAugmentation


def _page_tensor_cache_key(
    rel_path: str,
    img_height: int,
    max_width: Optional[int],
    max_text_chars: Optional[int],
    cache_namespace: str,
    *,
    u8_deferred: bool = False,
) -> str:
    rel = rel_path.replace("\\", "/")
    base = (cache_namespace or "", rel, img_height, max_width, max_text_chars)
    tup = base + ("u8pre_cuda",) if u8_deferred else base
    payload = repr(tup).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_document_text(txt: str, max_text_chars: Optional[int]) -> str:
    s = " ".join(txt.replace("\r\n", "\n").replace("\r", "\n").split())
    if max_text_chars is not None and max_text_chars > 0 and len(s) > int(max_text_chars):
        return s[: int(max_text_chars)]
    return s


def discover_page_txt_pairs(pair_root: Path) -> List[Tuple[Path, Path]]:
    """Ищем снимки вида *_page_image_*.png рядом с *_page_text_*.txt под pair_root или .../pages-img-plaintext."""
    roots: List[Path] = [pair_root]
    nested = pair_root / "pages-img-plaintext"
    if nested.is_dir():
        roots.append(nested)

    pairs: List[Tuple[Path, Path]] = []
    seen: set[str] = set()

    for root in roots:
        if not root.is_dir():
            continue
        for png in sorted(root.rglob("*.png")):
            stem = png.stem
            if "_image_" in stem:
                tstem = stem.replace("_image_", "_text_", 1)
            elif stem.endswith("_image"):
                tstem = stem[: -len("_image")] + "_text"
            else:
                continue
            txt_path = png.with_name(tstem + ".txt")
            if not txt_path.is_file():
                continue
            key = str(png.resolve())
            if key in seen:
                continue
            seen.add(key)
            pairs.append((png, txt_path))
    pairs.sort(key=lambda x: str(x[0]))
    return pairs


class PageTxtPairsDataset(Dataset):
    """Одно наблюдение = целиком страница (resize по высоте), текст из .txt после нормализации пробелов."""

    def __init__(
        self,
        pair_root: Union[str, Path],
        *,
        img_height: int = 32,
        max_width: Optional[int] = 1200,
        train_augmentation: Optional[TrainAugmentation] = None,
        preprocessed_cache_dir: Optional[Union[str, Path]] = None,
        preprocessed_ram_cache_max_bytes: Optional[int] = None,
        cache_namespace: str = "page_txt",
        max_text_chars: Optional[int] = None,
        defer_resize_normalize_to_cuda: bool = False,
        defer_png_bytes_to_collate: bool = False,
    ):
        self.pair_root = Path(pair_root).expanduser().resolve()
        self.samples = discover_page_txt_pairs(self.pair_root)
        if not self.samples:
            raise FileNotFoundError(
                f"page_txt_pairs: в {self.pair_root!s} не найдено ни одной пары PNG+TXT "
                f"вида *_page_image_*.png / *_page_text_*.txt (проверьте unzip pages-img-plaintext)."
            )
        self.img_height = img_height
        self.max_width = max_width
        self.train_augment = train_augmentation
        self._cache_namespace = (cache_namespace or "").strip()
        self.max_text_chars = max_text_chars
        self._defer_resize_cuda = bool(defer_resize_normalize_to_cuda)
        self._png_bytes_collate = bool(defer_png_bytes_to_collate)
        if self._png_bytes_collate and not self._defer_resize_cuda:
            raise ValueError("defer_png_bytes_to_collate требует defer_resize_normalize_to_cuda=True")

        rc = preprocessed_cache_dir
        self.preprocessed_cache_root: Optional[Path] = None
        if rc is not None:
            s = str(rc).strip()
            if s:
                self.preprocessed_cache_root = Path(s).expanduser().resolve()

        _rb = preprocessed_ram_cache_max_bytes
        self._ram_lru: Optional[_RamTensorLRU] = None
        if _rb is not None and int(_rb) > 0:
            self._ram_lru = _RamTensorLRU(int(_rb))

        if self._defer_resize_cuda and self.train_augment is not None:
            raise ValueError("defer_resize_normalize_to_cuda несовместим с augmentation_train для page_txt_pairs.")
        if self.preprocessed_cache_root is not None and self.train_augment is not None:
            raise ValueError("preprocessed_cache_dir несовместим с augmentation_train для page_txt_pairs.")
        if self._ram_lru is not None and self.train_augment is not None:
            raise ValueError("RAM-кэш несовместим с augmentation_train для page_txt_pairs.")

        self._rel_keys: List[str] = []
        for png, _ in self.samples:
            try:
                self._rel_keys.append(str(png.relative_to(self.pair_root)).replace("\\", "/"))
            except ValueError:
                self._rel_keys.append(str(png))

    def __len__(self) -> int:
        return len(self.samples)

    def _prepare_full_pil(self, pil: Image.Image) -> Tuple[torch.Tensor, int]:
        pil_rgb = pil.convert("RGB")
        if self.train_augment is not None:
            pil = self.train_augment(pil_rgb)
        else:
            pil = TF.rgb_to_grayscale(pil_rgb, num_output_channels=1)
        w_t, h_t = pil.size
        new_w = max(1, round(w_t * self.img_height / h_t))
        if self.max_width is not None and new_w > self.max_width:
            new_w = self.max_width
        crop_resized = pil.resize((new_w, self.img_height), Image.Resampling.BILINEAR)
        tens = TF.to_tensor(crop_resized)
        tens = TF.normalize(tens, mean=[0.5], std=[0.5])
        return tens, int(tens.shape[-1])

    def _tensor_tv_full(self, path: Path) -> Optional[Tuple[torch.Tensor, int]]:
        try:
            from torchvision.io import read_image
        except Exception:
            return None
        try:
            chw = read_image(str(path))
        except Exception:
            return None
        if chw.dtype != torch.uint8 or chw.dim() != 3:
            return None
        ch = int(chw.shape[0])
        if ch == 4:
            chw = chw[:3]
        elif ch not in (1, 3):
            return None
        x01 = chw.float().div_(255.0)
        if chw.shape[0] == 3:
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

    def _tensor_tv_uint8_ready(self, path: Path) -> Optional[Tuple[torch.Tensor, int]]:
        try:
            from torchvision.io import read_image
        except Exception:
            return None
        try:
            chw = read_image(str(path))
        except Exception:
            return None
        if chw.dtype != torch.uint8 or chw.dim() != 3:
            return None
        ch = int(chw.shape[0])
        if ch == 4:
            chw = chw[:3]
        elif ch not in (1, 3):
            return None
        x01 = chw.float().div_(255.0)
        if chw.shape[0] == 3:
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
        gu8 = (out.clamp(0.0, 1.0).mul_(255.0)).to(torch.uint8)
        return gu8, int(gu8.shape[-1])

    def _pil_uint8_ready(self, pil: Image.Image) -> Tuple[torch.Tensor, int]:
        pil_rgb = pil.convert("RGB")
        pil_g = TF.rgb_to_grayscale(pil_rgb, num_output_channels=1)
        w_t, h_t = pil_g.size
        new_w = max(1, round(w_t * self.img_height / h_t))
        if self.max_width is not None and new_w > self.max_width:
            new_w = self.max_width
        crop_resized = pil_g.resize((new_w, self.img_height), Image.Resampling.BILINEAR)
        arr = np.asarray(crop_resized, dtype=np.uint8)
        tens = torch.from_numpy(arr).unsqueeze(0).contiguous()
        return tens, int(tens.shape[-1])

    def _load_image_uint8(self, png_path: Path) -> Tuple[torch.Tensor, int]:
        if self.train_augment is None:
            tv = self._tensor_tv_uint8_ready(png_path)
            if tv is not None:
                return tv
        return self._pil_uint8_ready(Image.open(png_path))

    def _load_image_tensor(self, png_path: Path) -> Tuple[torch.Tensor, int]:
        if self.train_augment is None:
            tv = self._tensor_tv_full(png_path)
            if tv is not None:
                return tv
        pil = Image.open(png_path)
        return self._prepare_full_pil(pil)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        png_path, txt_path = self.samples[idx]
        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            raw_txt = f.read()
        text = _normalize_document_text(raw_txt, self.max_text_chars)
        rel_k = self._rel_keys[idx]
        ram = self._ram_lru
        cache_root = self.preprocessed_cache_root
        use_key = ram is not None or cache_root is not None

        key_hex = (
            _page_tensor_cache_key(
                rel_k,
                self.img_height,
                self.max_width,
                self.max_text_chars,
                self._cache_namespace,
                u8_deferred=self._defer_resize_cuda,
            )
            if use_key
            else ""
        )

        def pack_float(tens: torch.Tensor, w: int, tx: str) -> Dict[str, Union[torch.Tensor, str]]:
            return {
                LINE_PREP_KEY: LINE_PREP_TENSOR,
                "image": tens.contiguous(),
                "width": torch.tensor(w, dtype=torch.long),
                "text": tx,
            }

        def pack_u8(u8: torch.Tensor, w: int, tx: str) -> Dict[str, Union[torch.Tensor, str, int]]:
            return {
                LINE_PREP_KEY: LINE_PREP_UINT8,
                "image_u8": u8.contiguous(),
                "crop_h": int(self.img_height),
                "crop_w": int(w),
                U8_KIND_KEY: U8_KIND_PAGE_READY,
                "text": tx,
            }

        defer = self._defer_resize_cuda
        if defer:
            if ram is not None and key_hex:
                hit = ram.get_u8(key_hex)
                if hit is not None:
                    tens_u8, ch, cw = hit
                    if int(tens_u8.shape[-2]) == ch and int(tens_u8.shape[-1]) == cw:
                        return pack_u8(tens_u8, cw, text)

            def _take_u8_from_disk() -> Optional[Dict[str, Union[torch.Tensor, str, int]]]:
                u8t = disk_lines_try_u8_cache(cache_root, key_hex)
                if u8t is None:
                    return None
                payload, ch_, cw_ = u8t
                if ram is not None:
                    ram.put_u8(key_hex, payload, ch_, cw_)
                return pack_u8(payload, cw_, text)

            got = _take_u8_from_disk()
            if got is not None:
                return got
            if self._png_bytes_collate and self.train_augment is None:
                data = png_path.read_bytes()
                jb = torch.frombuffer(bytearray(data), dtype=torch.uint8)
                return {
                    LINE_PREP_KEY: LINE_PREP_PNG_PAGE,
                    "png_bytes": jb,
                    "text": text,
                }
            got2 = _take_u8_from_disk()
            if got2 is not None:
                return got2
            u8, out_w = self._load_image_uint8(png_path)
            if ram is not None and key_hex:
                _c, hh, ww = u8.shape
                ram.put_u8(key_hex, u8, int(hh), int(ww))
            if cache_root is not None and key_hex:
                hp = _lines_tensor_cache_path(cache_root, key_hex)
                _c2, hh2, ww2 = u8.shape
                try:
                    _save_lines_u8_cache(hp, u8, int(hh2), int(ww2))
                except Exception:
                    pass
            return pack_u8(u8, out_w, text)

        if ram is not None and key_hex:
            hitf = ram.get_float(key_hex)
            if hitf is not None:
                tens, out_w = hitf
                return pack_float(tens, out_w, text)

        def _take_float_from_disk() -> Optional[Dict[str, Union[torch.Tensor, str]]]:
            ft = disk_lines_try_float_cache(cache_root, key_hex)
            if ft is None:
                return None
            payload, wf = ft
            if ram is not None:
                ram.put_float(key_hex, payload, wf)
            return pack_float(payload, wf, text)

        gotf = _take_float_from_disk()
        if gotf is not None:
            return gotf

        tens, out_w = self._load_image_tensor(png_path)
        if ram is not None and key_hex:
            ram.put_float(key_hex, tens, out_w)
        if cache_root is not None and key_hex:
            hp = _lines_tensor_cache_path(cache_root, key_hex)
            try:
                _save_lines_tensor_cache(hp, tens, out_w)
            except Exception:
                pass
        return pack_float(tens, out_w, text)
