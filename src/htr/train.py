from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from htr.charset import Charset, charset_from_strings
from htr.data.coco_lines import COCOLinesDataset, coco_collate_fn
from htr.data.split import Subset, random_split_indices, texts_for_charset_from_coco
from htr.device import move_batch_to_device, pick_device
from htr.eval.metrics import lev_ratio
from htr.io.checkpoint import save_checkpoint
from htr.models import resolve_model
from htr.transforms import TrainAugmentation


def _pack_targets(texts: list[str], charset: Charset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = [charset.encode(t) for t in texts]
    tlens = torch.tensor([len(x) for x in tensors], dtype=torch.long, device=device)
    if tlens.sum().item() == 0:
        return torch.zeros(0, dtype=torch.long, device=device), tlens
    concat = torch.tensor([idx for seq in tensors for idx in seq], dtype=torch.long, device=device)
    return concat, tlens


def _reject_yaml_methodology_stubs(cfg: dict) -> None:
    name = str(cfg.get("model", {}).get("name", "")).lower()
    if not name.startswith("stub_section"):
        return
    mp = cfg.get("methodology_profile") if isinstance(cfg.get("methodology_profile"), dict) else {}
    sec = mp.get("report_section", "?")
    note_raw = mp.get("note", "")
    note = str(note_raw).strip() if note_raw is not None else ""
    tail = f" Примечание: {note}" if note else ""
    raise SystemExit(
        f"Конфигурация — YAML-заготовка §{sec} для отчёта; модель `{name}` пока не реализована в обучителе.{tail}"
    )


def run_training(cfg: dict) -> None:
    _reject_yaml_methodology_stubs(cfg)
    seed = int(cfg["project"]["seed"])
    torch.manual_seed(seed)

    dc = cfg["data"]
    train_aug = TrainAugmentation() if bool(dc.get("augmentation_train", False)) else None
    full_ds = COCOLinesDataset(
        coco_json=dc["coco_json"],
        image_root=dc["image_root"],
        text_field=dc.get("text_field", "translation"),
        img_height=int(dc["img_height"]),
        max_width=dc.get("max_width"),
        min_crop_width=int(dc.get("min_crop_width", 4)),
        train_augmentation=train_aug,
    )
    vf = float(dc.get("val_fraction", 0.0))
    train_ix, val_ix = random_split_indices(len(full_ds), vf, seed=seed)

    texts_train = texts_for_charset_from_coco(full_ds.samples, train_ix)
    extra = cfg.get("charset", {}).get("extra_chars") or ""
    charset = charset_from_strings(texts_train, extra)

    train_ds = Subset(full_ds, train_ix)

    eval_aug_backup = getattr(full_ds, "train_augment", None)
    full_ds.train_augment = None
    val_ds = Subset(full_ds, val_ix)
    full_ds.train_augment = eval_aug_backup

    from torch.utils.data import DataLoader

    nw = int(dc.get("num_workers", 0))
    bs = int(cfg["training"]["batch_size"])
    loader_train = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=bs,
        num_workers=nw,
        collate_fn=coco_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    loader_val = DataLoader(
        val_ds,
        shuffle=False,
        batch_size=max(1, bs // 2),
        num_workers=nw,
        collate_fn=coco_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    device_s = pick_device(cfg["project"].get("device", "cuda"))
    device = torch.device(device_s)

    model = resolve_model(cfg, charset.num_classes).to(device)

    criterion = nn.CTCLoss(blank=Charset.blank_idx, zero_infinity=True)
    lr = float(cfg["training"]["lr"])
    wd = float(cfg["training"].get("weight_decay", 0.0))
    optim_ = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    scaler = GradScaler(enabled=bool(cfg["training"].get("amp", False)))

    epochs = int(cfg["training"]["epochs"])
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    experiment = cfg["training"].get("experiment_name", "run")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        nb = 0
        bar = tqdm(loader_train, desc=f"epoch {epoch}/{epochs}", leave=False)
        for batch in bar:
            b = move_batch_to_device(batch, device)
            images = b["image"]  # type: ignore[arg-type]
            texts_batch: list[str] = b["text"]  # type: ignore[list-item]

            targets, target_lengths = _pack_targets(texts_batch, charset, device)

            optim_.zero_grad(set_to_none=True)
            use_amp = scaler.is_enabled()
            if use_amp:
                with torch.cuda.amp.autocast(enabled=True):
                    log_probs = model(images)
                    t_len = torch.full((images.shape[0],), log_probs.shape[0], dtype=torch.long, device=device)
                    loss = criterion(log_probs, targets, t_len, target_lengths)
                scaler.scale(loss).backward()
                clip = float(cfg["training"].get("clip_grad_norm", 5.0))
                if clip > 0:
                    scaler.unscale_(optim_)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optim_)
                scaler.update()
            else:
                log_probs = model(images)
                t_len = torch.full((images.shape[0],), log_probs.shape[0], dtype=torch.long, device=device)
                loss = criterion(log_probs, targets, t_len, target_lengths)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("clip_grad_norm", 5.0)))
                optim_.step()

            total_loss += float(loss.item())
            nb += 1
            bar.set_postfix(loss=loss.item())

        mean_loss = total_loss / max(1, nb)

        cer_sum = 0.0
        n_lab = 0
        model.eval()
        with torch.no_grad():
            if not val_ix:
                print(f"[epoch {epoch}] train_loss={mean_loss:.4f} (val: пусто val_fraction)")
            else:
                for batch in tqdm(loader_val, desc=f"val {epoch}", leave=False):
                    b = move_batch_to_device(batch, device)
                    images_b = b["image"]  # type: ignore[arg-type]
                    texts_ref: list[str] = b["text"]  # type: ignore[list-item]
                    lp = model(images_b)
                    greedy_seq = lp.argmax(dim=-1).transpose(0, 1).cpu().tolist()
                    for i, seq in enumerate(greedy_seq):
                        hyp = charset.decode_indices(seq)
                        _, ratio = lev_ratio(hyp, texts_ref[i])
                        cer_sum += ratio
                        n_lab += 1
                cer_val = cer_sum / max(1, n_lab)
                print(f"[epoch {epoch}] train_loss={mean_loss:.4f} val_sym_error_ratio={cer_val:.4f}")

        save_every = max(1, int(cfg["training"].get("save_every_epochs", 1)))
        if epoch % save_every == 0:
            path = ckpt_dir / f"{experiment}_e{epoch}.pt"
            save_checkpoint(str(path), model.state_dict(), itos=charset.itos, model_name=str(cfg["model"]["name"]), yaml_dump=dict(cfg))

    latest = ckpt_dir / "latest.pt"
    save_checkpoint(str(latest), model.state_dict(), itos=charset.itos, model_name=str(cfg["model"]["name"]), yaml_dump=dict(cfg))
