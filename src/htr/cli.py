from __future__ import annotations

import argparse
from pathlib import Path

from htr.config import config_paths_from_args, deep_merge, load_yaml


def train_main() -> None:
    from htr.train import run_training

    parser = argparse.ArgumentParser(description="Обучение строковой модели HTR/OCR.")
    parser.add_argument("--config", "-c", action="append", help="YAML-конфиг(и); второй перекрывает первый.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, dest="batch_size", default=None)
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default=None)
    args = parser.parse_args()

    paths = config_paths_from_args(default_name="default.yaml", config_arg=args.config)
    merged: dict = {}
    for p in paths:
        merged = deep_merge(merged, load_yaml(p))

    pr = merged.setdefault("project", {})
    if args.device is not None:
        pr["device"] = args.device
    tr = merged.setdefault("training", {})
    if args.epochs is not None:
        tr["epochs"] = args.epochs
    if args.batch_size is not None:
        tr["batch_size"] = args.batch_size

    merged["project"] = pr
    merged["training"] = tr

    run_training(merged)


def infer_main() -> None:
    from htr.infer import greedy_decode

    parser = argparse.ArgumentParser(description="Инференс по сохранённой строковой модели (crop одной линии).")
    parser.add_argument("--config", "-c", action="append", default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Приоритетнее, чем infer.checkpoint в YAML.")
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()

    ck_path = args.checkpoint
    merged: dict = {}
    if args.config is not None or ck_path is None:
        paths = config_paths_from_args(default_name="default.yaml", config_arg=args.config)
        merged = {}
        for p in paths:
            merged = deep_merge(merged, load_yaml(p))
        if ck_path is None:
            ck_path = merged.get("infer", {}).get("checkpoint")

    dh = merged.get("data", {}) if merged else {}
    img_h = int(dh.get("img_height", 32))
    max_w_raw = dh.get("max_width") if dh else None
    max_w = int(max_w_raw) if max_w_raw is not None else None

    if ck_path is None:
        raise SystemExit("укажите --checkpoint или infer.checkpoint и при необходимости --config")
    decoded = greedy_decode(Path(ck_path), [Path(i) for i in args.images], device_pref=args.device, img_height=img_h, max_width=max_w)
    for path, txt in zip(args.images, decoded):
        print(f"{path}\t{txt!r}")
