# practice

MTUCI Practice

## Данные для экспериментов

Объёмы набора данных в репозиторий не включены и перечислены в `.gitignore`. После клонирования проекта нужно один раз скачать архив и разложить каталоги, запустив скрипт из корня репозитория с параметром `fetch` (или синоним `download`):

```bash
./scripts/archive_data_datasets.sh fetch
```

Скрипт загрузит файл `archive.tar.gz` и распакует три папки в `data/digital_peter`, `data/yenisei_gov_reports_td`, `data/russian_old_orthography_ocr`. Рабочая директория при запуске не важна, если указан полный путь к скрипту.

Далее распакуйте вложенные ZIP внутри этих каталогов:

```bash
./scripts/unzip_datasets.sh
```

Параметр `pack` (или запуск без аргументов) делает обратную операцию: собирает локальные три папки в `practice_data_datasets.tar.gz` — нужен только при подготовке архива для обмена или облака, не для типичной настройки окружения.

## Строковый HTR-пайплайн (конфиг, обучение, инференс)

После установки данных пакет `htr` (каталог `src/htr`) даёт связку конфигурации YAML + CRNN со CTC без претрейна как базовый вариант архитектуры из поясняющего отчёта.

```bash
python -m pip install -e .
htr-train --config configs/default.yaml --config configs/baselines/4_01_line_crnn_ctc_digital_peter.yaml --epochs 3 --device cpu
htr-infer --checkpoint training/checkpoints/latest.pt path/to/crop_line.png
```

Эквивалент более короткой цепочки (алиас профилю §4.1):

```bash
htr-train --config configs/default.yaml --config configs/digital_peter_line_ctc.yaml --epochs 3 --device cpu
```

Без установки editable:

```bash
PYTHONPATH=src python -m htr.cli train_main --config configs/baselines/4_01_line_crnn_ctc_digital_peter.yaml --epochs 1 --device cpu
PYTHONPATH=src python -m htr.cli infer_main --checkpoint training/checkpoints/latest.pt path/to/crop_line.png
```

Чекпоинты по умолчанию в каталог `training/checkpoints/`; полный снимок YAML сохраняется внутрь `.pt` для воспроизводимости.
