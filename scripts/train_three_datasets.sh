#!/usr/bin/env bash
# Имя файла историческое: только Yenisei Gov Reports TD, как в configs/default.yaml (data.sources).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
exec htr-train --config "${ROOT}/configs/default.yaml" "$@"
