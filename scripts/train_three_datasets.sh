#!/usr/bin/env bash
# Имя файла историческое: configs/default.yaml — смесь двух COCO-корпусов (Yenisei + Digital Peter).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
exec htr-train --config "${ROOT}/configs/default.yaml" "$@"
