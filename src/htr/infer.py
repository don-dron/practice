from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, TextIO

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from htr.charset import Charset
from htr.device import pick_device
from htr.io.checkpoint import load_checkpoint
from htr.line_width_limits import clamp_line_width_px, line_resize_effective_height_px
from htr.models import resolve_model
from htr.models.attention_line import AttentionLineSeq2Seq


def _attn_hypothesis(ce_tokens: list[int], charset: Charset) -> str:
    out: List[str] = []
    for k in ce_tokens:
        cid = k + 1
        if 1 <= cid < len(charset.itos):
            out.append(charset.itos[cid])
    return "".join(out)


def _prepare_line_batch(
    pils: List[Image.Image],
    *,
    img_height: int,
    max_width: Optional[int],
    device: torch.device,
    line_resize_height_floor_px: Optional[float] = None,
    line_resize_height_cap_px: Optional[float] = None,
) -> torch.Tensor:
    seqs = []
    for pil_item in pils:
        pil_g = TF.rgb_to_grayscale(pil_item.convert("RGB"), num_output_channels=1)
        w_t, h_t = pil_g.size
        h_eff = line_resize_effective_height_px(
            float(h_t), floor_px=line_resize_height_floor_px, cap_px=line_resize_height_cap_px
        )
        new_w = clamp_line_width_px(w_t * img_height / h_eff, max_width=max_width)
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
    verbose: bool = True,
    log: TextIO = sys.stderr,
) -> List[str]:
    ckpt_path_resolved = checkpoint_pt.expanduser().resolve()
    ck = load_checkpoint(str(ckpt_path_resolved))
    charset_local = Charset.from_itos(list(ck["itos"]))
    merged_yaml: dict = {}
    yaml_raw = ck.get("config_yaml", {})
    if isinstance(yaml_raw, dict):
        merged_yaml = dict(yaml_raw)

    dh = merged_yaml.get("data", {})
    lr_floor_px: Optional[float] = None
    lr_cap_px: Optional[float] = None
    if dh:
        img_height = int(dh.get("img_height", img_height))
        mw_ck = dh.get("max_width")
        max_width = int(mw_ck) if mw_ck is not None else None

        rf = dh.get("line_resize_height_floor_px")
        if rf is not None:
            try:
                fv = float(rf)
            except (TypeError, ValueError):
                fv = 0.0
            if fv > 0:
                lr_floor_px = fv

        rc = dh.get("line_resize_height_cap_px")
        if rc is not None:
            try:
                cv = float(rc)
            except (TypeError, ValueError):
                cv = 0.0
            if cv > 0:
                lr_cap_px = cv

    merged_yaml.setdefault("model", {})
    merged_yaml["model"]["name"] = ck["model_name"]

    if verbose:
        ep = ck.get("completed_epoch")
        ep_s = str(ep) if ep is not None else "?"
        print(
            f"[htr-infer] checkpoint: {ckpt_path_resolved}",
            file=log,
        )
        print(
            f"[htr-infer] модель: {ck['model_name']} · классов (charset+метки): "
            f"{charset_local.num_classes} · сохранено после epoch: {ep_s}",
            file=log,
        )
        print(
            f"[htr-infer] ресайз ленты высотой {img_height}px; max_width={max_width}; "
            f"эффективная высота bbox h_eff floor={lr_floor_px} cap={lr_cap_px}",
            file=log,
        )

    net = resolve_model(merged_yaml, charset_local.num_classes)
    net.load_state_dict(ck["state_dict"])
    dev_torch = torch.device(pick_device(device_pref))
    net.to(dev_torch)
    net.eval()

    tram = merged_yaml.get("training")
    tram_d = tram if isinstance(tram, dict) else {}
    attn_cap = max(16, int(tram_d.get("decoder_max_steps", 512)))

    pils_loaded = [Image.open(p_el) for p_el in image_paths]

    def _pil_size(pil: Image.Image) -> tuple[int, int]:
        w_t, h_t = pil.size
        return int(w_t), int(h_t)

    if verbose:
        for ip, pil_item in zip(image_paths, pils_loaded):
            w_i, h_i = _pil_size(pil_item)
            h_eff = line_resize_effective_height_px(
                float(h_i), floor_px=lr_floor_px, cap_px=lr_cap_px
            )
            new_w_est = clamp_line_width_px(
                float(w_i) * float(img_height) / h_eff,
                max_width=max_width,
            )
            tag = ""
            if w_i > max(600, img_height * 45):
                tag = " (похоже на целую страницу — режим только «техдемо», распознанный текст обычно бессмыслица)"
            print(
                f"[htr-infer] файл: {ip} · исходно {w_i}×{h_i}px → ресайз в ленту ≈ {new_w_est}×{img_height}{tag}",
                file=log,
            )

    bat_t = _prepare_line_batch(
        pils_loaded,
        img_height=img_height,
        max_width=max_width,
        device=dev_torch,
        line_resize_height_floor_px=lr_floor_px,
        line_resize_height_cap_px=lr_cap_px,
    )

    if verbose:
        print(
            f"[htr-infer] тензор на {bat_t.device}: shape={tuple(bat_t.shape)} dtype={bat_t.dtype} "
            "(декод: greedy из log_probs, без beam)",
            file=log,
        )

    results_txt: List[str] = []

    with torch.no_grad():
        if isinstance(net, AttentionLineSeq2Seq):
            pred_lists = net.greedy_inference(bat_t, attn_cap)
            results_txt = [_attn_hypothesis(pl, charset_local) for pl in pred_lists]
        else:
            log_probs_mat = net(bat_t)
            greedy_list = log_probs_mat.argmax(dim=-1).transpose(0, 1).cpu().tolist()
            if verbose and greedy_list:
                seq0 = greedy_list[0]
                cnt = Counter(seq0)
                tops = cnt.most_common(6)

                def _lbl(i: int) -> str:
                    if i == charset_local.blank_idx:
                        return "<blk>"
                    if 0 <= i < len(charset_local.itos):
                        c = charset_local.itos[i]
                        r = repr(c)
                        return r if len(r) <= 24 else r[:21] + "…"
                    return f"#{i}"

                tlen = len(seq0)
                blanks = cnt.get(charset_local.blank_idx, 0)
                hdr = [(f"{_lbl(k)}->{v}") for k, v in tops]
                print(
                    f"[htr-infer] greedy по времени [batch 0]: T={tlen}; <blk> на {blanks}/{tlen} шагах; топ: [{', '.join(hdr)}]",
                    file=log,
                )
                if blanks >= int(0.85 * tlen):
                    print(
                        "[htr-infer] подсказка: доминирует <blk> — по CTC модель почти не отдаёт буквы; так бывает при "
                        "малом числе эпох/подвыборке train и любом качестве. Улучшают: меньше train_subsample_divisor, "
                        "больше epochs, второй конфиг gpu_large если хватает VRAM, инференс по crop строки:",
                        "`PYTHONPATH=src python scripts/infer_one_coco_line.py --checkpoint … --file 9.jpg`",
                        file=log,
                    )
                dominant = tops[0] if tops else None
                if dominant and dominant[0] != charset_local.blank_idx and dominant[1] > max(10, int(0.55 * tlen)):
                    print(
                        "[htr-infer] подсказка: почти один и тот же не-blank символ на большинстве шагов — модель «залипла» или "
                        "вход похож во всех картинках (напр. ресайз целой страницы без crop). Обучите дольше/на строках; "
                        "для проверки используйте crop из COCO: scripts/infer_one_coco_line.py",
                        file=log,
                    )

            for row_ids in greedy_list:
                results_txt.append(charset_local.decode_indices(row_ids))

    if verbose:
        print("[htr-infer] — итог (ниже строка path<TAB>текст для парсинга; при машинном разборе используйте --quiet) —", file=log)
        for ip, txt in zip(image_paths, results_txt):
            print(f"[htr-infer]   {Path(ip).name}: {txt!r} (символов: {len(txt)})", file=log)

    return results_txt
