#!/usr/bin/env bash
# Упаковать локальные data/* в practice_data_datasets.tar.gz:
#   ./scripts/archive_data_datasets.sh pack
# или по умолчанию:
#   ./scripts/archive_data_datasets.sh
#
# Скачать архив с Яндекс.Диска и разложить в data/digital_peter, … (в архиве в корне три папки):
#   ./scripts/archive_data_datasets.sh fetch
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PUBLIC_FOLDER_KEY="https://disk.yandex.ru/d/dXi_n8w5Q-U5iA"
META_URL="https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=${PUBLIC_FOLDER_KEY}"
DOWNLOAD_ARCHIVE="$ROOT/archive.tar.gz"

pack() {
	local ARCHIVE="$ROOT/practice_data_datasets.tar.gz"
	cd "$ROOT"
	echo "Создание: $ARCHIVE"
	echo "(это несколько гигабайт — подождите)"
	COPYFILE_DISABLE=1 tar czf "$ARCHIVE" \
		data/digital_peter \
		data/yenisei_gov_reports_td \
		data/russian_old_orthography_ocr
	ls -lh "$ARCHIVE"
	echo "Готово (pack)."
}

fetch() {
	cd "$ROOT"
	echo "Загрузка с Яндекс.Диска → $DOWNLOAD_ARCHIVE"
	# Получить прямую ссылку и скачать (как в вашей команде curl + sed).
	# shellcheck disable=SC2016
	local href
	href="$(
		curl -fsS "$META_URL" |
			sed -n 's/.*"href":"\([^"]*\)".*/\1/p' |
			sed 's/\\u0026/\&/g'
	)"
	if [[ -z "$href" ]]; then
		echo "Ошибка: не удалось извлечь href из ответа API Яндекс.Диска." >&2
		exit 1
	fi

	curl -fL "$href" -o "$DOWNLOAD_ARCHIVE"

	local TMP
	TMP="$(mktemp -d "${TMPDIR:-/tmp}/practice_data_unpack.XXXXXX")"
	cleanup_tmp() {
		rm -rf "$TMP"
	}
	trap cleanup_tmp EXIT

	echo "Распаковка архива во временную папку…"
	tar xzf "$DOWNLOAD_ARCHIVE" -C "$TMP"

	# Корень архива может быть напрямую из трёх папок или одна обёртка поверх них.
	local BASE="$TMP"
	if [[ ! -d "$TMP/digital_peter" || ! -d "$TMP/russian_old_orthography_ocr" || ! -d "$TMP/yenisei_gov_reports_td" ]]; then
		BASE=""
		for cand in "$TMP"/*; do
			if [[ -d "$cand/digital_peter" && -d "$cand/russian_old_orthography_ocr" && -d "$cand/yenisei_gov_reports_td" ]]; then
				BASE="$cand"
				break
			fi
		done
	fi
	if [[ -z "${BASE:-}" ]]; then
		echo "Ошибка: в архиве нет набора из трёх каталогов (ни в корне tarball, ни в одной подпапке)." >&2
		ls -la "$TMP"
		exit 1
	fi

	mkdir -p "$ROOT/data"
	for d in digital_peter russian_old_orthography_ocr yenisei_gov_reports_td; do
		if [[ ! -d "$BASE/$d" ]]; then
			echo "Ошибка: нет \"$BASE/$d\" после распаковки." >&2
			exit 1
		fi
		rm -rf "$ROOT/data/$d"
		mv "$BASE/$d" "$ROOT/data/"
	done

	trap - EXIT
	cleanup_tmp

	ls -lah "$DOWNLOAD_ARCHIVE"
	echo "Готово (fetch): \$ROOT/data/{digital_peter,russian_old_orthography_ocr,yenisei_gov_reports_td}"
	echo "При необходимости распакуйте внутренние ZIP: ./scripts/unzip_datasets.sh"
}

usage() {
	echo "Использование: $0 [pack|fetch]" >&2
	echo "  pack  — собрать practice_data_datasets.tar.gz из каталогов data/" >&2
	echo "  fetch — скачать archive.tar.gz с Яндекс.Диска и разложить три папки в data/" >&2
	exit 2
}

case "${1:-pack}" in
	pack) pack ;;
	fetch | download) fetch ;;
	-h | --help | help) usage ;;
	*)
		echo "Неизвестная команда: $1" >&2
		usage
		;;
esac
