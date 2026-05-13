@echo off
setlocal EnableDelayedExpansion
REM Full registry section 5.1 sequence: T0, E1, E1*, E2, S2..S5.
REM From repo root:  scripts\train_registry_all.cmd --device cuda
REM Snapshots: training\registry_snapshots\<tag>_latest.pt
REM
REM Python: prefers .venv\Scripts\python.exe ; override with TRAIN_REGISTRY_PYTHON=full\path\to\python.exe
REM If torch has no CUDA but you ask for cuda: reinstalls GPU wheels ^(needs internet^).
REM Skip auto pip fix: set TRAIN_REGISTRY_SKIP_CUDA_PIP=1
REM Allow CPU-only run without pip: set TRAIN_REGISTRY_ALLOW_CPU=1 ^|^| pass --device cpu

cd /d "%~dp0.."
set "REPO_ROOT=%CD%"
set "PYTHONPATH=%REPO_ROOT%\src"

REM ---- Interpreter: .venv preferred ----
set "EXE_PY="
if exist "%REPO_ROOT%\.venv\Scripts\python.exe" set "EXE_PY=%REPO_ROOT%\.venv\Scripts\python.exe"
if defined TRAIN_REGISTRY_PYTHON set "EXE_PY=%TRAIN_REGISTRY_PYTHON%"
if not defined EXE_PY set "EXE_PY=python"

REM ---- Optional: create .venv ----
if not exist "%REPO_ROOT%\.venv\Scripts\python.exe" (
  where py >nul 2>&1
  if not errorlevel 1 (
    echo [train_registry] Creating .venv with py launcher...
    py -3 -m venv "%REPO_ROOT%\.venv"
  ) else (
    "%EXE_PY%" -m venv "%REPO_ROOT%\.venv" 2>nul
  )
  if exist "%REPO_ROOT%\.venv\Scripts\python.exe" set "EXE_PY=%REPO_ROOT%\.venv\Scripts\python.exe"
)

set "SNAP=%REPO_ROOT%\training\registry_snapshots"
if not exist "%SNAP%" mkdir "%SNAP%"

set "OPTS="
if defined TRAIN_REGISTRY_DEVICE set "OPTS=!OPTS! --device !TRAIN_REGISTRY_DEVICE!"
if defined TRAIN_REGISTRY_EPOCHS set "OPTS=!OPTS! --epochs !TRAIN_REGISTRY_EPOCHS!"
if defined TRAIN_REGISTRY_BATCH set "OPTS=!OPTS! --batch-size !TRAIN_REGISTRY_BATCH!"

set "USER=%*"
set "ALL=!OPTS!"
if not "!USER!"=="" set "ALL=!ALL! !USER!"
if "!ALL!"=="" set "ALL=--device cuda"

REM ---- Pip / CUDA bootstrap for GPU runs ----
echo !ALL! | findstr /i /c:"--device cpu" >nul
if errorlevel 1 (
  REM default path: wants GPU
  if not defined TRAIN_REGISTRY_SKIP_CUDA_PIP (
    call :ensure_torch_cuda "!EXE_PY!"
    if errorlevel 1 (
      echo [train_registry] CUDA bootstrap ^(pip / torch^) failed - see messages above.
      exit /b 4
    )
  )
  "!EXE_PY!" -c "import torch; import sys; print('torch.cuda.is_available=', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)" 2>nul
  if errorlevel 1 (
    if not defined TRAIN_REGISTRY_ALLOW_CPU (
      echo ERROR: CUDA still not available after bootstrap. Set TRAIN_REGISTRY_SKIP_CUDA_PIP=0 removed^? Try manual pip from pytorch.org
      exit /b 3
    )
    echo WARNING: continuing on CPU ^(TRAIN_REGISTRY_ALLOW_CPU=1^).
  )
)

echo.
echo PYTHONPATH=%PYTHONPATH%
echo Python: "!EXE_PY!"
echo Extra train flags: !ALL!
echo.

echo ========== T0 ==========
"!EXE_PY!" -c "import sys; from htr.cli import train_main; sys.argv=['htr-train']+sys.argv[1:]; train_main()" --config configs\default.yaml !ALL!
if errorlevel 1 goto :err
copy /Y "%REPO_ROOT%\training\checkpoints\latest.pt" "%SNAP%\T0_latest.pt" >nul

echo ========== E1 ==========
"!EXE_PY!" -c "import sys; from htr.cli import train_main; sys.argv=['htr-train']+sys.argv[1:]; train_main()" --config configs\default.yaml --config configs\baselines\4_01_line_crnn_ctc_digital_peter.yaml !ALL!
if errorlevel 1 goto :err
copy /Y "%REPO_ROOT%\training\checkpoints\latest.pt" "%SNAP%\E1_latest.pt" >nul

echo ========== E1_star ==========
"!EXE_PY!" -c "import sys; from htr.cli import train_main; sys.argv=['htr-train']+sys.argv[1:]; train_main()" --config configs\default.yaml --config configs\digital_peter_line_ctc.yaml !ALL!
if errorlevel 1 goto :err
copy /Y "%REPO_ROOT%\training\checkpoints\latest.pt" "%SNAP%\E1_star_latest.pt" >nul

echo ========== E2 ==========
"!EXE_PY!" -c "import sys; from htr.cli import train_main; sys.argv=['htr-train']+sys.argv[1:]; train_main()" --config configs\default.yaml --config configs\baselines\overlay_comparable_line_budget.yaml --config configs\baselines\4_01_line_crnn_ctc_digital_peter.yaml !ALL!
if errorlevel 1 goto :err
copy /Y "%REPO_ROOT%\training\checkpoints\latest.pt" "%SNAP%\E2_latest.pt" >nul

echo ========== S2 ==========
"!EXE_PY!" -c "import sys; from htr.cli import train_main; sys.argv=['htr-train']+sys.argv[1:]; train_main()" --config configs\default.yaml --config configs\baselines\4_02_line_stub_encoder_decoder_attention.yaml !ALL!
if errorlevel 1 goto :err
copy /Y "%REPO_ROOT%\training\checkpoints\latest.pt" "%SNAP%\S2_latest.pt" >nul

echo ========== S3 ==========
"!EXE_PY!" -c "import sys; from htr.cli import train_main; sys.argv=['htr-train']+sys.argv[1:]; train_main()" --config configs\default.yaml --config configs\baselines\4_03_line_stub_transformer_encoder_attention.yaml !ALL!
if errorlevel 1 goto :err
copy /Y "%REPO_ROOT%\training\checkpoints\latest.pt" "%SNAP%\S3_latest.pt" >nul

echo ========== S4 ==========
"!EXE_PY!" -c "import sys; from htr.cli import train_main; sys.argv=['htr-train']+sys.argv[1:]; train_main()" --config configs\default.yaml --config configs\baselines\4_04_page_stub_detector_then_line.yaml !ALL!
if errorlevel 1 goto :err
copy /Y "%REPO_ROOT%\training\checkpoints\latest.pt" "%SNAP%\S4_latest.pt" >nul

echo ========== S5 ^(first run may download ResNet weights^) ==========
"!EXE_PY!" -c "import sys; from htr.cli import train_main; sys.argv=['htr-train']+sys.argv[1:]; train_main()" --config configs\default.yaml --config configs\baselines\4_05_line_stub_transfer_pretrained_encoder.yaml !ALL!
if errorlevel 1 goto :err
copy /Y "%REPO_ROOT%\training\checkpoints\latest.pt" "%SNAP%\S5_latest.pt" >nul

echo.
echo Done. Snapshots under %SNAP%
dir /b "%SNAP%"
exit /b 0

REM ---------------------------------------------------------------------------
REM %1 = python.exe path ; exit 0 if cuda ok after steps, 1 if hard fail
:ensure_torch_cuda
set "_PY=%~1"
echo [train_registry] Using: "!_PY!"
"!_PY!" -m pip install -q -U pip
if errorlevel 1 (
  echo [train_registry] pip upgrade failed.
  exit /b 1
)

"!_PY!" -m pip install -q -e "%REPO_ROOT%"
if errorlevel 1 (
  echo [train_registry] pip install -e . failed ^(offline^? deps^?)
  exit /b 1
)

"!_PY!" -c "import sys; sys.exit(0 if __import__('torch').cuda.is_available() else 1)" 2>nul
if not errorlevel 1 exit /b 0

echo [train_registry] torch without working CUDA - reinstalling GPU wheels (cu124, then cu121). Needs internet.

"!_PY!" -m pip uninstall -y torch torchvision torchaudio 2>nul
"!_PY!" -m pip install --upgrade-strategy eager torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
  echo [train_registry] cu124 wheel install failed ^(network/url^?)
)
"!_PY!" -c "import sys; sys.exit(0 if __import__('torch').cuda.is_available() else 1)" 2>nul
if not errorlevel 1 (
  echo [train_registry] OK: CUDA after cu124
  exit /b 0
)

echo [train_registry] Retrying CUDA wheels with cu121...
"!_PY!" -m pip uninstall -y torch torchvision torchaudio 2>nul
"!_PY!" -m pip install --upgrade-strategy eager torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
  echo [train_registry] cu121 wheel install failed.
  exit /b 1
)
"!_PY!" -c "import sys; sys.exit(0 if __import__('torch').cuda.is_available() else 1)" 2>nul
if not errorlevel 1 (
  echo [train_registry] OK: CUDA after cu121
  exit /b 0
)
echo [train_registry] Still no CUDA from PyTorch builds on this interpreter.
exit /b 1

:err
echo Train step failed, exit code %ERRORLEVEL%.
exit /b 1
