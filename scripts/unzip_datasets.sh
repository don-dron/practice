#!/usr/bin/env bash
# Распаковать внутренние ZIP в каталогах data/ (работает из любой директории).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/data"

unzip_to() {
	local zipfile="$1"
	local dest_parent="$2"
	if [[ ! -f "$zipfile" ]]; then
		echo "Пропуск (нет файла): $zipfile" >&2
		return 0
	fi
	echo "Распаковка: $zipfile → $dest_parent/"
	unzip -oq "$zipfile" -d "$dest_parent"
}

unzip_to "$DATA/digital_peter/images.zip" "$DATA/digital_peter"
unzip_to "$DATA/yenisei_gov_reports_td/test_images.zip" "$DATA/yenisei_gov_reports_td"
unzip_to "$DATA/yenisei_gov_reports_td/train_images.zip" "$DATA/yenisei_gov_reports_td"
unzip_to "$DATA/russian_old_orthography_ocr/books-pdf-plaintext.zip" "$DATA/russian_old_orthography_ocr"
unzip_to "$DATA/russian_old_orthography_ocr/pages-img-plaintext.zip" "$DATA/russian_old_orthography_ocr"

rm -rf "$DATA/digital_peter/__MACOSX"

echo "Готово. Пути: data/digital_peter/images, yenisei_*/{test_,train_}images, russian_old_orthography_ocr/{books-pdf-plaintext,pages-img-plaintext}."
