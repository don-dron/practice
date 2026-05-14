from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from htr.charset import Charset
from htr.device import pick_device
from htr.io.checkpoint import load_checkpoint
from htr.line_width_limits import clamp_line_width_px
from htr.models import resolve_model
from htr.models.attention_line import AttentionLineSeq2Seq


def _attn_hypothesis(ce_tokens: list[int], charset: Charset) -> str:
    out: List[str] = []
    for k in ce_tokens:
        cid = k + 1
        if 1 <= cid < len(charset.itos):
            out.append(charset.itos[cid])
    return "".join(out)


def _prepare_line_batch(pils: List[Image.Image], *, img_height: int, max_width: Optional[int], device: torch.device) -> torch.Tensor:
    seqs = []
    for pil_item in pils:
        pil_g = TF.rgb_to_grayscale(pil_item.convert("RGB"), num_output_channels=1)
        w_t, h_t = pil_g.size
        new_w = clamp_line_width_px(w_t * img_height / h_t, max_width=max_width)
        resized = pil_g.resize((new_w, img_height), Image.Resampling.BILINEAR)
        tens = TF.normalize(TF.to_tensor(resized), mean=[0.5], std=[0.5])
        seqs.append(tens)
    bw_max = max(s.shape[-1] for s in seqs)
    batch_t = torch.zeros(len(seqs), 1, img_height, bw_max, dtype=seqs[0].dtype, device=device)
    for i_b, tens_a in enumerate(seqs):
        w_sub = tens_a.shape[-1]
        batch_t[i_b, :, :, :w_sub] = tens_a.to(device)
    return batch_t


def greedy_decode(
    checkpoint_pt: Path,
    image_paths: List[Path],
    *,
    device_pref: str = "cuda",
    img_height: int = 32,
    max_width: Optional[int] = 1200,
) -> List[str]:
    ck = load_checkpoint(str(checkpoint_pt))
    charset_local = Charset.from_itos(list(ck["itos"]))
    merged_yaml: dict = {}
    yaml_raw = ck.get("config_yaml", {})
    if isinstance(yaml_raw, dict):
        merged_yaml = dict(yaml_raw)

    dh = merged_yaml.get("data", {})
    if dh:
        img_height = int(dh.get("img_height", img_height))
        mw_ck = dh.get("max_width")
        max_width = int(mw_ck) if mw_ck is not None else None

    merged_yaml.setdefault("model", {})
    merged_yaml["model"]["name"] = ck["model_name"]

    net = resolve_model(merged_yaml, charset_local.num_classes)
    net.load_state_dict(ck["state_dict"])
    dev_torch = torch.device(pick_device(device_pref))
    net.to(dev_torch)
    net.eval()

    tram = merged_yaml.get("training")
    tram_d = tram if isinstance(tram, dict) else {}
    attn_cap = max(16, int(tram_d.get("decoder_max_steps", 512)))

    pils_loaded = [Image.open(p_el) for p_el in image_paths]
    bat_t = _prepare_line_batch(pils_loaded, img_height=img_height, max_width=max_width, device=dev_torch)

    results_txt: List[str] = []

    with torch.no_grad():
        if isinstance(net, AttentionLineSeq2Seq):
            pred_lists = net.greedy_inference(bat_t, attn_cap)
            results_txt = [_attn_hypothesis(pl, charset_local) for pl in pred_lists]
        else:
            log_probs_mat = net(bat_t)
            greedy_list = log_probs_mat.argmax(dim=-1).transpose(0, 1).cpu().tolist()
            for row_ids in greedy_list:
                results_txt.append(charset_local.decode_indices(row_ids))
    return results_txt
