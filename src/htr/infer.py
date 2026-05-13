from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torchvision.transforms.functional as TF
from PIL import Image
import torch

from htr.charset import Charset
from htr.device import pick_device
from htr.io.checkpoint import load_checkpoint
from htr.models import resolve_model


def _prepare_line_batch(pils: List[Image.Image], *, img_height: int, max_width: Optional[int], device: torch.device) -> torch.Tensor:
    seqs = []
    for pil in pils:
        pil = TF.rgb_to_grayscale(pil.convert("RGB"), num_output_channels=1)
        w_t, h_t = pil.size
        new_w = max(1, round(w_t * img_height / h_t))
        if max_width is not None and new_w > max_width:
            new_w = max_width
        pil_resized = pil.resize((new_w, img_height), Image.Resampling.BILINEAR)
        tens = TF.normalize(TF.to_tensor(pil_resized), mean=[0.5], std=[0.5])
        seqs.append(tens)
    bw = max(s.shape[-1] for s in seqs)
    batch = torch.zeros(len(seqs), 1, img_height, bw, dtype=seqs[0].dtype, device=device)
    for i, s in enumerate(seqs):
        w = s.shape[-1]
        batch[i, :, :, :w] = s.to(device)
    return batch


def greedy_decode(
    checkpoint_pt: Path,
    image_paths: List[Path],
    *,
    device_pref: str = "cuda",
    img_height: int = 32,
    max_width: Optional[int] = 1200,
) -> List[str]:
    ck = load_checkpoint(str(checkpoint_pt))
    charset = Charset.from_itos(list(ck["itos"]))
    model_cfg_yaml = ck.get("config_yaml", {})
    merged = dict(model_cfg_yaml) if isinstance(model_cfg_yaml, dict) else {}
    dh_ck = merged.get("data", {})
    if dh_ck:
        img_height = int(dh_ck.get("img_height", img_height))
        mw = dh_ck.get("max_width")
        max_width = int(mw) if mw is not None else None

    merged.setdefault("model", {})
    merged["model"]["name"] = ck["model_name"]

    model = resolve_model(merged, charset.num_classes)
    model.load_state_dict(ck["state_dict"])
    dev = torch.device(pick_device(device_pref))
    model.to(dev)
    model.eval()

    pils = [Image.open(p) for p in image_paths]
    bat = _prepare_line_batch(pils, img_height=img_height, max_width=max_width, device=dev)

    out: List[str] = []
    with torch.no_grad():
        lp = model(bat)
        greedy_seq = lp.argmax(dim=-1).transpose(0, 1).cpu().tolist()
        for seq in greedy_seq:
            out.append(charset.decode_indices(seq))
    return out
