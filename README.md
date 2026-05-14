# practice

MTUCI Practice

## Данные для экспериментов

Объёмы набора данных в репозиторий не включены и перечислены в `.gitignore`. После клонирования проекта нужно один раз скачать архив и разложить каталоги, запустив скрипт из корня репозитория с параметром `fetch` (или синоним `download`):

```bash
./scripts/archive_data_datasets.sh fetch
```

Скрипт загрузит файл `archive.tar.gz` и распакует в `data/yenisei_gov_reports_td` только этот каталог (в архиве с облака могут быть лишние папки — они не переносятся). Рабочая директория при запуске не важна, если указан полный путь к скрипту.

Скачивание и распаковка на **Windows** из `cmd.exe` или PowerShell:

```bat
scripts\archive_data_datasets.cmd fetch
scripts\unzip_datasets.cmd
```

Для `fetch` нужен однократный вызов `powershell.exe`, встроенного в систему — это только строка получения JSON с Яндекс.Диска; отдельные `.ps1` в репозитории не используются.

Далее один раз распакуйте вложенные ZIP: на Unix — `./scripts/unzip_datasets.sh`, на Windows — `scripts\unzip_datasets.cmd` (ZIP через `Expand-Archive` в PowerShell; большой `tar.gz` из `fetch` распаковывается системным `%SystemRoot%\System32\tar.exe`, не `tar` из Git). Перенос во `data\` после `fetch` — `robocopy`.

Параметр `pack` (или запуск без аргументов) делает обратную операцию: собирает локальный `data/yenisei_gov_reports_td` в `practice_data_datasets.tar.gz` — нужен только при подготовке архива для обмена или облака, не для типичной настройки окружения.

## Строковый HTR-пайплайн (конфиг, обучение, инференс)

После установки данных пакет `htr` (каталог `src/htr`) даёт связку конфигурации YAML + CRNN со CTC без претрейна как базовый вариант архитектуры из поясняющего отчёта. **По умолчанию только Yenisei Gov Reports TD** (`data/yenisei_gov_reports_td/...`, `text_field: transcription`). Другой корпус: задайте `data.sources` или `coco_json` в своём YAML.

```bash
python -m pip install -e .
htr-train --config configs/default.yaml --epochs 2 --device cpu
htr-infer --checkpoint training/checkpoints/latest.pt path/to/crop_line.png
```

Дополнительный YAML — другая архитектура или гиперпараметры; пути к данным во втором файле проверяйте перед запуском:

```bash
htr-train --config configs/default.yaml --config configs/baselines/4_01_line_crnn_ctc_digital_peter.yaml --epochs 2 --device cpu
```

Без установки editable (пример только Yenisei):

```bash
PYTHONPATH=src python -c "import sys; from htr.cli import train_main; sys.argv=['htr-train']+sys.argv[1:]; train_main()" --config configs/default.yaml --epochs 2 --device cpu
PYTHONPATH=src python -m htr.cli infer_main --checkpoint training/checkpoints/latest.pt path/to/crop_line.png
```

Чекпоинты по умолчанию в каталог `training/checkpoints/`; полный снимок YAML сохраняется внутрь `.pt` для воспроизводимости.
