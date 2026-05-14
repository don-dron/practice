#!/usr/bin/env bash
# Быстро: размер кэша предобработанных строк (du). Медленно: число .pt под деревом.
# Запуск из корня репозитория: ./scripts/disk_cache_stats.sh [training/preprocessed_cache]
set -euo pipefail
ROOT="${1:-training/preprocessed_cache}"
if [[ ! -d "$ROOT" ]]; then
  echo "Нет каталога: $ROOT (сначала задайте data.preprocessed_cache_dir в YAML и запустите train хотя бы частично)."
  exit 1
fi
echo "Размеры (du -sh):"
du -sh "$ROOT" 2>/dev/null || true
if [[ -d "$ROOT" ]] && [[ -n "$(ls -A "$ROOT" 2>/dev/null)" ]]; then
  du -sh "$ROOT"/* 2>/dev/null || true
fi
echo ""
echo "Подсчёт всех .pt под $ROOT (на больших кэшах — минуты)..."
find "$ROOT" -name '*.pt' -print 2>/dev/null | wc -l | tr -d ' '
echo "ожидаемое число ≈ сумме примеров по корпусам (см. лог htr-train при старте)."
