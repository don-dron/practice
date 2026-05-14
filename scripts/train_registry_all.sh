#!/usr/bin/env bash
# Последовательно обучает все связки регистра §5.1 (T0, E1, E1*, E2, S2–S5).
# Все шаги используют data.sources из configs/default.yaml (Yenisei по умолчанию), если второй YAML не переопределяет sources целиком.
# Запуск из корня репозитория: ./scripts/train_registry_all.sh [--epochs N] [--batch-size N] [--device cuda|cpu|mps]
#
# По умолчанию эпохи и batch из YAML.
# Из окружения: TRAIN_REGISTRY_DEVICE, TRAIN_REGISTRY_EPOCHS, TRAIN_REGISTRY_BATCH
# Полный YAML-бюджет (долго): ./scripts/train_registry_all.sh --device cuda
# Быстрый дым: TRAIN_REGISTRY_EPOCHS=1 ./scripts/train_registry_all.sh --device cpu --batch-size 4
#
# После каждого прогона копируется training/checkpoints/latest.pt в
# training/registry_snapshots/<метка>_latest.pt

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export PYTHONPATH="${REPO}/src"
PYTHON="${PYTHON:-python3}"

SNAP_DIR="$REPO/training/registry_snapshots"
mkdir -p "$SNAP_DIR"

run_train_copy() {
  local tag="$1"
  shift
  echo ""
  echo "========== регистр: $tag =========="
  "$PYTHON" -c "
import sys
from htr.cli import train_main

sys.argv = ['htr-train'] + sys.argv[1:]
train_main()
" "$@"
  local dest="$SNAP_DIR/${tag}_latest.pt"
  cp -f "$REPO/training/checkpoints/latest.pt" "$dest"
  echo "snapshot: $dest"
}



ENV_ARGS=()
[[ -n "${TRAIN_REGISTRY_DEVICE:-}" ]] && ENV_ARGS+=( --device "${TRAIN_REGISTRY_DEVICE}" )
[[ -n "${TRAIN_REGISTRY_EPOCHS:-}" ]] && ENV_ARGS+=( --epochs "${TRAIN_REGISTRY_EPOCHS}" )
[[ -n "${TRAIN_REGISTRY_BATCH:-}" ]] && ENV_ARGS+=( --batch-size "${TRAIN_REGISTRY_BATCH}" )

COMMON_EXTRA=( "${ENV_ARGS[@]}" "$@" )

run_train_copy T0 --config configs/default.yaml "${COMMON_EXTRA[@]}"

run_train_copy E1 --config configs/default.yaml --config configs/baselines/4_01_line_crnn_ctc_digital_peter.yaml "${COMMON_EXTRA[@]}"

run_train_copy E1_star --config configs/default.yaml --config configs/digital_peter_line_ctc.yaml "${COMMON_EXTRA[@]}"

run_train_copy E2 --config configs/default.yaml --config configs/baselines/overlay_comparable_line_budget.yaml --config configs/baselines/4_01_line_crnn_ctc_digital_peter.yaml "${COMMON_EXTRA[@]}"

run_train_copy S2 --config configs/default.yaml --config configs/baselines/4_02_line_stub_encoder_decoder_attention.yaml "${COMMON_EXTRA[@]}"

run_train_copy S3 --config configs/default.yaml --config configs/baselines/4_03_line_stub_transformer_encoder_attention.yaml "${COMMON_EXTRA[@]}"

run_train_copy S4 --config configs/default.yaml --config configs/baselines/4_04_page_stub_detector_then_line.yaml "${COMMON_EXTRA[@]}"

run_train_copy S5 --config configs/default.yaml --config configs/baselines/4_05_line_stub_transfer_pretrained_encoder.yaml "${COMMON_EXTRA[@]}"

echo ""
echo "Готово: снимки в $SNAP_DIR"
ls -la "$SNAP_DIR"
