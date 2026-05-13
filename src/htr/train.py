from __future__ import annotations

from pathlib import Path
import os
import sys

import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm

from htr.charset import Charset, charset_from_strings
from htr.data.coco_lines import COCOLinesDataset, coco_collate_fn
from htr.data.split import Subset, random_split_indices, texts_for_charset_from_coco
from htr.device import move_batch_to_device, pick_device
from htr.eval.metrics import lev_ratio
from htr.io.checkpoint import save_checkpoint
from htr.models import resolve_model
from htr.models.attention_line import AttentionLineSeq2Seq
from htr.models.resnet_pretrained_line_ctc import PretrainedResnetLineCTC
from htr.transforms import TrainAugmentation


def _make_grad_scaler(enabled: bool):
    """GradScaler: torch.amp API on newer PyTorch, legacy import on older builds."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    from torch.cuda.amp import GradScaler as _LegacyGradScaler

    return _LegacyGradScaler(enabled=enabled)


def _autocast_cuda():
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=True)
    import torch.cuda.amp as cuda_amp

    return cuda_amp.autocast(enabled=True)


def _pack_targets(texts_batch: list[str], charset: Charset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = [charset.encode(t) for t in texts_batch]
    tlens = torch.tensor([len(x) for x in tensors], dtype=torch.long, device=device)
    if tlens.sum().item() == 0:
        return torch.zeros(0, dtype=torch.long, device=device), tlens
    concat = torch.tensor([idx for seq in tensors for idx in seq], dtype=torch.long, device=device)
    return concat, tlens


def _training_objective(cfg: dict) -> str:
    return str(cfg.get("training", {}).get("objective", "ctc")).strip().lower()


def _decoder_max_steps(cfg: dict) -> int:
    return max(16, int(cfg.get("training", {}).get("decoder_max_steps", 512)))


def _freeze_from_plan(cfg: dict) -> int:
    ppt = cfg.get("planned_transfer_policy")
    if not isinstance(ppt, dict):
        return 0
    fx = ppt.get("freeze_encoder_until_epoch")
    if fx is None:
        return 0
    try:
        return max(0, int(fx))
    except (TypeError, ValueError):
        return 0


def _freeze_backbone_epochs(cfg: dict) -> int:
    mh = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    fv = mh.get("freeze_backbone_epochs")
    if fv is not None:
        try:
            return max(0, int(fv))
        except (TypeError, ValueError):
            return _freeze_from_plan(cfg)
    return _freeze_from_plan(cfg)


def _set_pretrained_backbone_frozen(model: nn.Module, backbone_frozen: bool) -> None:
    if not isinstance(model, PretrainedResnetLineCTC):
        return
    for p in model.backbone_parameters():
        p.requires_grad = not backbone_frozen


def _truncate_enc(enc: list[int], cap_steps: int) -> list[int]:
    steps = len(enc) + 1 if enc else 1
    if steps <= cap_steps:
        return enc
    return enc[: max(0, cap_steps - 1)]


def _prepare_attention_batches(
    model: AttentionLineSeq2Seq,
    charset: Charset,
    texts_batch: list[str],
    device: torch.device,
    decoder_cap: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sos = model.sos_idx_emb
    pad_emb = model.pad_idx_emb
    eos_c = model.eos_logits

    batch_enc: list[list[int]] = []
    lengths = []
    for tex in texts_batch:
        ee = charset.encode(tex)
        ee_t = _truncate_enc(ee, decoder_cap)
        batch_enc.append(ee_t)
        ln = len(ee_t)
        lengths.append(1 if ln == 0 else ln + 1)

    lm = max(lengths)
    b_sz = len(texts_batch)

    tin = torch.full((b_sz, lm), pad_emb, dtype=torch.long, device=device)
    targ = torch.full((b_sz, lm), -100, dtype=torch.long, device=device)
    msk = torch.zeros(b_sz, lm, dtype=torch.bool, device=device)

    for bi, enc_ids in enumerate(batch_enc):
        if not enc_ids:
            msk[bi, 0] = True
            tin[bi, 0] = sos
            targ[bi, 0] = eos_c
            continue

        steps = len(enc_ids) + 1
        for s in range(steps):
            msk[bi, s] = True
            tin[bi, s] = sos if s == 0 else enc_ids[s - 1] - 1
            if s < len(enc_ids):
                targ[bi, s] = enc_ids[s] - 1
            else:
                targ[bi, s] = eos_c

    return tin, targ, msk


def _attention_hypothesis(ce_tokens: list[int], charset: Charset) -> str:
    out: list[str] = []
    for k in ce_tokens:
        cid = k + 1
        if 1 <= cid < len(charset.itos):
            out.append(charset.itos[cid])
    return "".join(out)


def _dataload_worker_count(dc: dict) -> int:
    nw_requested = int(dc.get("num_workers", 0))
    nw = max(0, nw_requested)
    if sys.platform == "win32":
        _cap_raw = os.environ.get("HTR_WIN_MAX_NUM_WORKERS", "").strip()
        if _cap_raw != "0":
            _cap = 16
            if _cap_raw != "":
                try:
                    _cap = max(1, min(32, int(_cap_raw)))
                except ValueError:
                    pass
            if nw > _cap:
                print(
                    f"[htr-train] num_workers: YAML asked {nw}, capping at {_cap} on Windows "
                    f"(raise: larger YAML num_workers / set HTR_WIN_MAX_NUM_WORKERS=N; disable cap: =0)"
                )
                nw = _cap
    return nw


def _preprocessed_ram_budget(dc: dict, nw: int) -> tuple[int, float]:
    raw = dc.get("preprocessed_ram_cache_max_gb", 6.0)
    if raw is None:
        gb = 6.0
    else:
        try:
            gb = float(raw)
        except (TypeError, ValueError):
            gb = 6.0
    if gb <= 0:
        return 0, gb
    total = gb * (1024.0**3)
    if nw <= 0:
        return max(int(total), 1024), gb
    denom = max(1, nw) * 2
    return max(int(total / denom), 1024), gb


def run_training(cfg: dict) -> None:
    objective = _training_objective(cfg)
    decoder_cap = _decoder_max_steps(cfg)
    seed = int(cfg["project"]["seed"])
    torch.manual_seed(seed)

    dc = cfg["data"]
    nw = _dataload_worker_count(dc)
    ram_budget_b, ram_gb_yaml = _preprocessed_ram_budget(dc, nw)

    train_aug = TrainAugmentation() if bool(dc.get("augmentation_train", False)) else None
    full_ds = COCOLinesDataset(
        coco_json=dc["coco_json"],
        image_root=dc["image_root"],
        text_field=dc.get("text_field", "translation"),
        img_height=int(dc["img_height"]),
        max_width=dc.get("max_width"),
        min_crop_width=int(dc.get("min_crop_width", 4)),
        train_augmentation=train_aug,
        preprocessed_cache_dir=dc.get("preprocessed_cache_dir"),
        preprocessed_ram_cache_max_bytes=(ram_budget_b if ram_budget_b > 0 else None),
    )
    if full_ds.preprocessed_cache_root is not None:
        print(f"[htr-train] preprocessed_cache_dir={full_ds.preprocessed_cache_root!s}")
    if ram_budget_b > 0:
        split_d = max(1, nw) * (2 if nw > 0 else 1)
        print(
            f"[htr-train] preprocessed_ram_cache_max_gb(total)≈{ram_gb_yaml:g} · "
            f"~{ram_budget_b / (1024**3):.3f} GiB per DataLoader worker "
            f"(split /{split_d}: train + val loaders × num_workers; num_workers={nw})"
        )
    vf = float(dc.get("val_fraction", 0.0))
    train_ix, val_ix = random_split_indices(len(full_ds), vf, seed=seed)

    texts_train = texts_for_charset_from_coco(full_ds.samples, train_ix)
    extra = cfg.get("charset", {}).get("extra_chars") or ""
    charset = charset_from_strings(texts_train, extra)

    train_ds = Subset(full_ds, train_ix)
    bak = getattr(full_ds, "train_augment", None)
    full_ds.train_augment = None
    val_ds = Subset(full_ds, val_ix)
    full_ds.train_augment = bak

    from torch.utils.data import DataLoader

    bs = int(cfg["training"]["batch_size"])

    _pin_memory = torch.cuda.is_available()
    _dl_common: dict = {
        "num_workers": nw,
        "collate_fn": coco_collate_fn,
        "pin_memory": _pin_memory,
    }
    if nw > 0:
        _dl_common["persistent_workers"] = True
        try:
            _pf = max(2, min(16, int(dc.get("prefetch_factor", 4))))
        except (TypeError, ValueError):
            _pf = 4
        _dl_common["prefetch_factor"] = _pf

    loader_train = DataLoader(train_ds, shuffle=True, batch_size=bs, **_dl_common)
    loader_val = DataLoader(
        val_ds,
        shuffle=False,
        batch_size=max(1, bs // 2),
        **_dl_common,
    )
    if nw > 0:
        print(
            f"[htr-train] dataloader workers={nw} prefetch_factor={_dl_common.get('prefetch_factor')} "
            "persistent_workers=true"
        )

    device_pref = cfg["project"].get("device", "cuda")
    resolved = pick_device(device_pref)
    device = torch.device(resolved)

    tc = cfg.get("training") if isinstance(cfg.get("training"), dict) else {}
    if device.type == "cuda":
        if tc.get("cudnn_benchmark", True):
            torch.backends.cudnn.benchmark = True
        if tc.get("allow_tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    print(
        f"[htr-train] device: requested={device_pref!r} -> {device!r} "
        f"(torch.cuda.is_available={torch.cuda.is_available()})"
    )
    if device_pref == "cuda" and not torch.cuda.is_available():
        print(
            "[htr-train] WARNING: CUDA requested but not available; training on CPU. "
            "Install GPU build: https://pytorch.org/get-started/locally/"
        )
    model = resolve_model(cfg, charset.num_classes).to(device)

    freeze_ep = _freeze_backbone_epochs(cfg)

    criterion = nn.CTCLoss(blank=Charset.blank_idx, zero_infinity=True) if objective != "attention_ce" else None
    lr = float(cfg["training"]["lr"])
    wd = float(cfg["training"].get("weight_decay", 0.0))

    if freeze_ep > 0:
        _set_pretrained_backbone_frozen(model, True)

    optim_ = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    amp_cfg = bool(cfg["training"].get("amp", False))
    scaler = _make_grad_scaler(enabled=(amp_cfg and device.type == "cuda"))

    epochs = int(cfg["training"]["epochs"])
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    experiment = cfg["training"].get("experiment_name", "run")

    for epoch in range(1, epochs + 1):
        if freeze_ep > 0 and isinstance(model, PretrainedResnetLineCTC):
            _set_pretrained_backbone_frozen(model, epoch <= freeze_ep)

        model.train()
        total_loss = 0.0
        nb_tr = 0
        train_bar = tqdm(loader_train, desc=f"epoch {epoch}/{epochs}", leave=False)
        for batch in train_bar:
            b_t = move_batch_to_device(batch, device)
            images = b_t["image"]  # type: ignore[arg-type]
            texts_batch: list[str] = b_t["text"]  # type: ignore[list-item]

            optim_.zero_grad(set_to_none=True)
            use_amp = scaler.is_enabled()

            if objective == "attention_ce":
                if not isinstance(model, AttentionLineSeq2Seq):
                    raise TypeError("objective attention_ce требует model.name attention_line_seq2seq")
                tin_y, targ_y, m_ok = _prepare_attention_batches(model, charset, texts_batch, device, decoder_cap)
                if use_amp:
                    with _autocast_cuda():
                        loss = model.compute_loss_ce(images, tin_y, targ_y, m_ok)
                    scaler.scale(loss).backward()
                    clip_val = float(cfg["training"].get("clip_grad_norm", 5.0))
                    if clip_val > 0:
                        scaler.unscale_(optim_)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                    scaler.step(optim_)
                    scaler.update()
                else:
                    loss = model.compute_loss_ce(images, tin_y, targ_y, m_ok)
                    loss.backward()
                    clip_val = float(cfg["training"].get("clip_grad_norm", 5.0))
                    if clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                    optim_.step()

            else:
                assert criterion is not None
                targets_tl, tgt_lengths = _pack_targets(texts_batch, charset, device)
                if use_amp:
                    with _autocast_cuda():
                        log_probs = model(images)
                        inp_len = torch.full((images.shape[0],), log_probs.shape[0], dtype=torch.long, device=device)
                        batch_loss = criterion(log_probs, targets_tl, inp_len, tgt_lengths)
                    scaler.scale(batch_loss).backward()
                    clip_val = float(cfg["training"].get("clip_grad_norm", 5.0))
                    if clip_val > 0:
                        scaler.unscale_(optim_)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                    scaler.step(optim_)
                    scaler.update()
                else:
                    log_probs = model(images)
                    inp_len = torch.full((images.shape[0],), log_probs.shape[0], dtype=torch.long, device=device)
                    batch_loss = criterion(log_probs, targets_tl, inp_len, tgt_lengths)
                    batch_loss.backward()
                    clip_val = float(cfg["training"].get("clip_grad_norm", 5.0))
                    if clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                    optim_.step()

                loss = batch_loss

            total_loss += float(loss.item())
            nb_tr += 1
            train_bar.set_postfix(loss=loss.item())

        mean_train = total_loss / max(1, nb_tr)
        cer_sum = 0.0
        n_lab = 0

        model.eval()
        with torch.no_grad():
            if not val_ix:
                print(f"[epoch {epoch}] train_loss={mean_train:.4f} (val: пусто val_fraction)")
            else:
                for vbatch in tqdm(loader_val, desc=f"val {epoch}", leave=False):
                    b_v = move_batch_to_device(vbatch, device)
                    imgs_b = b_v["image"]  # type: ignore[arg-type]
                    refs_txt: list[str] = b_v["text"]  # type: ignore[list-item]
                    if objective == "attention_ce" and isinstance(model, AttentionLineSeq2Seq):
                        ce_preds = model.greedy_inference(imgs_b, decoder_cap)
                        for bi, sq in enumerate(ce_preds):
                            hyp_txt = _attention_hypothesis(sq, charset)
                            _, ratio_val = lev_ratio(hyp_txt, refs_txt[bi])
                            cer_sum += ratio_val
                            n_lab += 1
                    else:
                        log_p = model(imgs_b)
                        greedy_sequences = log_p.argmax(dim=-1).transpose(0, 1).cpu().tolist()
                        for bi, sq in enumerate(greedy_sequences):
                            hyp_txt = charset.decode_indices(sq)
                            _, ratio_val = lev_ratio(hyp_txt, refs_txt[bi])
                            cer_sum += ratio_val
                            n_lab += 1
                avg_cer = cer_sum / max(1, n_lab)
                print(f"[epoch {epoch}] train_loss={mean_train:.4f} val_sym_error_ratio={avg_cer:.4f}")

        save_every = max(1, int(cfg["training"].get("save_every_epochs", 1)))
        if epoch % save_every == 0:
            epath = ckpt_dir / f"{experiment}_e{epoch}.pt"
            save_checkpoint(
                str(epath),
                model.state_dict(),
                itos=charset.itos,
                model_name=str(cfg["model"]["name"]),
                yaml_dump=dict(cfg),
            )

    latest_ck = ckpt_dir / "latest.pt"
    save_checkpoint(
        str(latest_ck),
        model.state_dict(),
        itos=charset.itos,
        model_name=str(cfg["model"]["name"]),
        yaml_dump=dict(cfg),
    )
