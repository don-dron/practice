#!/usr/bin/env python3
"""Смок-тест checkpoint'ов из training/registry_snapshots (greedy decode на одном или нескольких изображениях).

Из корня репозитория:
  PYTHONPATH=src python scripts/registry_snapshots_smoke.py --device cpu
  python scripts/registry_snapshots_smoke.py -c checkpoints/T0_latest.pt path/to/crop_line.jpg

По умолчанию берёт все training/registry_snapshots/*_latest.pt и до трёх JPEG из train_images для быстрой проверки.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_path() -> None:
    repo = _repo_root()
    src = repo / "src"
    sp = str(src)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def _default_snapshots_dir(repo: Path) -> Path:
    return repo / "training" / "registry_snapshots"


def _default_sample_images(repo: Path, *, limit: int) -> list[Path]:
    imgs = repo / "data" / "yenisei_gov_reports_td" / "train_images"
    if not imgs.is_dir():
        return []
    out: list[Path] = []
    for patt in ("*.jpg", "*.jpeg", "*.png"):
        for p in sorted(imgs.glob(patt)):
            out.append(p)
            if len(out) >= limit:
                return out
    return out


def main() -> int:
    _ensure_src_path()
    repo = _repo_root()

    p = argparse.ArgumentParser(description="Смок registry_snapshots: greedy decode каждого *_latest.pt")
    p.add_argument(
        "--snap-dir",
        type=Path,
        default=None,
        help="Каталог со snapshots (по умолчанию training/registry_snapshots)",
    )
    p.add_argument("--device", type=str, default="cpu", choices=("cuda", "cpu", "mps"))
    p.add_argument(
        "images",
        nargs="*",
        help="Пути к crop-линиям / фрагментам; если пусто — авто JPEG из yenisei train_images",
    )
    p.add_argument("--limit-snapshots", type=int, default=0, help="Только первые N .pt (0 = все)")
    p.add_argument("--limit-images", type=int, default=3, help="Сколько картинок из датасета по умолчанию")
    args = p.parse_args()

    snap_dir = (args.snap_dir or _default_snapshots_dir(repo)).expanduser().resolve()
    if not snap_dir.is_dir():
        print(f"[smoke] нет каталога: {snap_dir}", file=sys.stderr)
        return 2

    ckpts = sorted(snap_dir.glob("*_latest.pt"))
    if not ckpts:
        print(f"[smoke] нет файлов *_latest.pt в {snap_dir}", file=sys.stderr)
        return 2

    if args.limit_snapshots > 0:
        ckpts = ckpts[: args.limit_snapshots]

    if args.images:
        image_paths = [Path(x).expanduser().resolve() for x in args.images]
    else:
        image_paths = _default_sample_images(repo, limit=max(1, args.limit_images))
        if not image_paths:
            print(
                "[smoke] не переданы изображения и не найден data/yenisei_gov_reports_td/train_images\n"
                "  Пример: python scripts/registry_snapshots_smoke.py --device cpu crop1.jpg crop2.jpg",
                file=sys.stderr,
            )
            return 2

    for ip in image_paths:
        if not ip.is_file():
            print(f"[smoke] нет файла: {ip}", file=sys.stderr)
            return 2

    from htr.infer import greedy_decode

    print(f"[smoke] snapshots: {len(ckpts)} · images: {len(image_paths)} · device={args.device!r}")

    failures = 0
    for ck in ckpts:
        tag = ck.stem.removesuffix("_latest")
        row_ok: list[str] = []
        try:
            texts = greedy_decode(ck, image_paths, device_pref=args.device, verbose=False)
            for img_p, txt in zip(image_paths, texts):
                short = repr(txt[:80] + ("…" if len(txt) > 80 else ""))
                row_ok.append(f"{img_p.name}:{short}")
            print(f"OK   {tag:12} {' | '.join(row_ok)}")
        except Exception as ex:
            failures += 1
            tb = "".join(traceback.format_exception_only(type(ex), ex)).strip()
            print(f"FAIL {tag:12} {tb}", file=sys.stderr)

    if failures:
        print(f"[smoke] упало загрузкой/infer: {failures}/{len(ckpts)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
