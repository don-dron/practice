@echo off
setlocal EnableDelayedExpansion
REM Full registry §5.1 sequence: T0, E1, E1*, E2, S2..S5.
REM Run from repo root:
REM   scripts\train_registry_all.cmd --device cuda
REM Smoke:
REM   set TRAIN_REGISTRY_EPOCHS=1
REM   scripts\train_registry_all.cmd --device cuda --batch-size 4
REM Snapshots: training\registry_snapshots\<tag>_latest.pt

cd /d "%~dp0.."
set "PYTHONPATH=%CD%\src"

set "SNAP=%CD%\training\registry_snapshots"
if not exist "%SNAP%" mkdir "%SNAP%"

set "OPTS="
if defined TRAIN_REGISTRY_DEVICE set "OPTS=!OPTS! --device !TRAIN_REGISTRY_DEVICE!"
if defined TRAIN_REGISTRY_EPOCHS set "OPTS=!OPTS! --epochs !TRAIN_REGISTRY_EPOCHS!"
if defined TRAIN_REGISTRY_BATCH set "OPTS=!OPTS! --batch-size !TRAIN_REGISTRY_BATCH!"

set "USER=%*"
set "ALL=!OPTS!"
if not "!USER!"=="" set "ALL=!ALL! !USER!"
if "!ALL!"=="" set "ALL=--device cuda"

REM CUDA: stop early unless CPU was requested explicitly or ALLOW_CPU override
python -c "import torch; import sys; print('torch.cuda.is_available=', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)" 2>nul
if errorlevel 1 (
  echo !ALL! | findstr /i /c:"--device cpu" >nul
  if errorlevel 1 (
    if not defined TRAIN_REGISTRY_ALLOW_CPU (
      echo ERROR: PyTorch has no CUDA ^(torch.cuda.is_available=False^).
      echo Fix: pip uninstall torch torchvision torchaudio -y
      echo Then install GPU wheels from https://pytorch.org/get-started/locally/
      echo Example CUDA 12.1: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
      echo CPU bypass: set TRAIN_REGISTRY_ALLOW_CPU=1 before running ^^|^| pass --device cpu
      exit /b 2
    )
    echo WARNING: CUDA missing; continuing on CPU ^(TRAIN_REGISTRY_ALLOW_CPU=1^).
  )
)

set PY=python

echo PYTHONPATH=%PYTHONPATH%
echo Extra train flags: !ALL!
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

echo ========== S5 ^(first run may download ResNet weights^) ==========
%PY% -c "import sys; from htr.cli import train_main; sys.argv=['htr-train']+sys.argv[1:]; train_main()" --config configs\default.yaml --config configs\baselines\4_05_line_stub_transfer_pretrained_encoder.yaml !ALL!
if errorlevel 1 goto :err
copy /Y "%CD%\training\checkpoints\latest.pt" "%SNAP%\S5_latest.pt" >nul

echo.
echo Done. Snapshots under %SNAP%
dir /b "%SNAP%"
exit /b 0

:err
echo Train step failed, exit code %ERRORLEVEL%.
exit /b 1
