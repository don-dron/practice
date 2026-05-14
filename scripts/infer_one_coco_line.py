#!/usr/bin/env python3
"""Один crop строки из Yenisei COCO + greedy infer (как при обучении, не целая страница).

Из корня репозитория:
  PYTHONPATH=src python scripts/infer_one_coco_line.py \\
    --checkpoint training/registry_snapshots/T0_latest.pt \\
    --coco data/yenisei_gov_reports_td/train.json \\
    --image-root data/yenisei_gov_reports_td/train_images \\
    --file 9.jpg --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))

    p = argparse.ArgumentParser(description="Infer по одной строке из COCO (bbox crop).")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--coco", type=Path, default=repo / "data/yenisei_gov_reports_td/train.json")
    p.add_argument("--image-root", type=Path, default=repo / "data/yenisei_gov_reports_td/train_images")
    p.add_argument("--file", required=True, help="file_name в COCO (например 9.jpg)")
    p.add_argument("--line-index", type=int, default=0, help="какая аннотация по порядку для этой страницы (0=первая)")
    p.add_argument("--device", default="cpu", choices=("cuda", "cpu", "mps"))
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    coco = json.loads(args.coco.read_text(encoding="utf-8"))
    id_by_name = {im["file_name"]: im["id"] for im in coco["images"]}
    if args.file not in id_by_name:
        print(f"нет file_name={args.file!r} в {args.coco}", file=sys.stderr)
        return 2
    iid = id_by_name[args.file]
    lines: list[dict] = [a for a in coco["annotations"] if a["image_id"] == iid]
    if not lines:
        print(f"нет аннотаций для image_id={iid}", file=sys.stderr)
        return 2
    li = min(max(0, args.line_index), len(lines) - 1)
    ann = lines[li]
    bbox = ann["bbox"]
    if len(bbox) != 4:
        print("bbox ожидался [x,y,w,h]", file=sys.stderr)
        return 2
    x, y, w, h = (float(b) for b in bbox)
    ref = (ann.get("attributes") or {}).get("transcription", "")
    img_path = args.image_root / args.file
    if not img_path.is_file():
        print(f"нет файла {img_path}", file=sys.stderr)
        return 2

    from PIL import Image

    pil = Image.open(img_path).convert("RGB")
    W, H = pil.size
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(W, int(x + w))
    y1 = min(H, int(y + h))
    crop = pil.crop((x0, y0, x1, y1))

    from htr.infer import greedy_decode

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        outp = Path(tf.name)
    try:
        crop.save(outp)
        res = greedy_decode(
            args.checkpoint,
            [outp],
            device_pref=args.device,
            verbose=not args.quiet,
        )[0]
    finally:
        outp.unlink(missing_ok=True)

    if not args.quiet:
        print(f"[coco-line] страница={args.file} line_index={li} bbox_xywh={[x,y,w,h]!r}")
        print(f"[coco-line] размер crop: {crop.size[0]}×{crop.size[1]} px")
        print(f"[coco-line] эталон (transcription): {ref!r}")

    print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
