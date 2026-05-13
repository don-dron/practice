@echo off
setlocal EnableDelayedExpansion
REM Все связки регистра §5.1 подряд: T0, E1, E1*, E2, S2..S5.
REM Запуск из корня репозитория (двойной клик тоже возможен, если рабочий каталог настроен):
REM   scripts\train_registry_all.cmd --device cuda
REM Быстрый дым:
REM   set TRAIN_REGISTRY_EPOCHS=1
REM   scripts\train_registry_all.cmd --device cuda --batch-size 4
REM Снимки копируются в training\registry_snapshots\<метка>_latest.pt

cd /d "%~dp0.."
set "PYTHONPATH=%CD%\src"

python -c "import torch; import sys; print('torch.cuda.is_available=', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)" 2>nul
if errorlevel 1 echo WARNING: PyTorch has no CUDA. Install GPU build from pytorch.org ^(pick CUDA matching your driver^). Training falls back to CPU.
echo.

set "SNAP=%CD%\training\registry_snapshots"
if not exist "%SNAP%" mkdir "%SNAP%"

REM Опции из переменных среды (как на bash)
set "OPTS="
if defined TRAIN_REGISTRY_DEVICE set "OPTS=!OPTS! --device !TRAIN_REGISTRY_DEVICE!"
if defined TRAIN_REGISTRY_EPOCHS set "OPTS=!OPTS! --epochs !TRAIN_REGISTRY_EPOCHS!"
if defined TRAIN_REGISTRY_BATCH set "OPTS=!OPTS! --batch-size !TRAIN_REGISTRY_BATCH!"

REM Аргументы командной строки дополняют OPTS; если всё пусто — по умолчанию CUDA
set "USER=%*"
set "ALL=!OPTS!"
if not "!USER!"=="" set "ALL=!ALL! !USER!"
if "!ALL!"=="" set "ALL=--device cuda"

set PY=python

echo PYTHONPATH=%PYTHONPATH%
echo Общие флаги: !ALL!
echo.

echo ========== T0 ==========
%PY% -c "import sys; from htr.cli import train_main; sys.argv=['htr-train']+sys.argv[1:]; train_main()" --config configs\default.yaml !ALL!
if errorlevel 1 goto :err
copy /Y "%CD%\training\checkpoints\latest.pt" "%SNAP%\T0_latest.pt" >nul

echo ========== E1 ==========
%PY% -c "import sys; from htr.cli import train_main; sys.argv=['htr-train']+sys.argv[1:]; train_main()" --config configs\default.yaml --config configs\baselines\4_01_line_crnn_ctc_digital_peter.yaml !ALL!
if errorlevel 1 goto :err
copy /Y "%CD%\training\checkpoints\latest.pt" "%SNAP%\E1_latest.pt" >nul

echo ========== E1_star ==========
%PY% -c "import sys; from htr.cli import train_main; sys.argv=['htr-train']+sys.argv[1:]; train_main()" --config configs\default.yaml --config configs\digital_peter_line_ctc.yaml !ALL!
if errorlevel 1 goto :err
copy /Y "%CD%\training\checkpoints\latest.pt" "%SNAP%\E1_star_latest.pt" >nul

echo ========== E2 ==========
%PY% -c "import sys; from htr.cli import train_main; sys.argv=['htr-train']+sys.argv[1:]; train_main()" --config configs\default.yaml --config configs\baselines\overlay_comparable_line_budget.yaml --config configs\baselines\4_01_line_crnn_ctc_digital_peter.yaml !ALL!
if errorlevel 1 goto :err
copy /Y "%CD%\training\checkpoints\latest.pt" "%SNAP%\E2_latest.pt" >nul

echo ========== S2 ==========
%PY% -c "import sys; from htr.cli import train_main; sys.argv=['htr-train']+sys.argv[1:]; train_main()" --config configs\default.yaml --config configs\baselines\4_02_line_stub_encoder_decoder_attention.yaml !ALL!
if errorlevel 1 goto :err
copy /Y "%CD%\training\checkpoints\latest.pt" "%SNAP%\S2_latest.pt" >nul

echo ========== S3 ==========
%PY% -c "import sys; from htr.cli import train_main; sys.argv=['htr-train']+sys.argv[1:]; train_main()" --config configs\default.yaml --config configs\baselines\4_03_line_stub_transformer_encoder_attention.yaml !ALL!
if errorlevel 1 goto :err
copy /Y "%CD%\training\checkpoints\latest.pt" "%SNAP%\S3_latest.pt" >nul

echo ========== S4 ==========
%PY% -c "import sys; from htr.cli import train_main; sys.argv=['htr-train']+sys.argv[1:]; train_main()" --config configs\default.yaml --config configs\baselines\4_04_page_stub_detector_then_line.yaml !ALL!
if errorlevel 1 goto :err
copy /Y "%CD%\training\checkpoints\latest.pt" "%SNAP%\S4_latest.pt" >nul

echo ========== S5 (первый раз может скачать веса ResNet) ==========
%PY% -c "import sys; from htr.cli import train_main; sys.argv=['htr-train']+sys.argv[1:]; train_main()" --config configs\default.yaml --config configs\baselines\4_05_line_stub_transfer_pretrained_encoder.yaml !ALL!
if errorlevel 1 goto :err
copy /Y "%CD%\training\checkpoints\latest.pt" "%SNAP%\S5_latest.pt" >nul

echo.
echo Готово. Снимки: %SNAP%
dir /b "%SNAP%"
exit /b 0

:err
echo Ошибка обучения, код %ERRORLEVEL%.
exit /b 1
