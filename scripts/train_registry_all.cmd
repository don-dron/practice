@echo off
setlocal EnableDelayedExpansion
REM Full registry §5.1: T0, E1, E1*, E2, S2 to S5. Data: merged from default.yaml sources (typically 2 COCO datasets).
REM From repo root:  scripts\train_registry_all.cmd --device cuda
REM Snapshots: training\registry_snapshots\<tag>_latest.pt
REM
REM Python: prefers .venv\Scripts\python.exe ; override with TRAIN_REGISTRY_PYTHON=full\path\to\python.exe
REM If torch has no CUDA but you ask for cuda: reinstalls GPU wheels ^(needs internet^).
REM Skip auto pip fix: set TRAIN_REGISTRY_SKIP_CUDA_PIP=1
REM Allow CPU-only run without pip: set TRAIN_REGISTRY_ALLOW_CPU=1 ^|^| pass --device cpu
REM Quieter pip: set TRAIN_REGISTRY_PIP_QUIET=1  ^(adds -q^)
REM Exit codes from :ensure_torch_cuda: 0 OK, 1 pip/venv failure, 2 CUDA false after wheels

cd /d "%~dp0"
cd /d ".."
set "REPO_ROOT=%CD%"
set "PYTHONPATH=%REPO_ROOT%\src"
set "TRPROBE=%~dp0train_registry_torch_probe.py"

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

set "CLI_ARGS=%*"
set "ALL=!OPTS!"
if not "!CLI_ARGS!"=="" set "ALL=!ALL! !CLI_ARGS!"
if "!ALL!"=="" set "ALL=--device cuda"

REM ---- Pip / CUDA bootstrap for GPU runs ----
REM Flat flow (no nested IF parens): avoids fragile cmd parsers on some locales / PowerShell-invoked CMD.
echo !ALL! | findstr /i /c:"--device cpu" >nul
if errorlevel 1 goto gpu_wants_cuda
goto after_gpu_cuda_bootstrap

:gpu_wants_cuda
if defined TRAIN_REGISTRY_SKIP_CUDA_PIP goto after_torch_continue
call :ensure_torch_cuda "!EXE_PY!"
if errorlevel 2 goto tr_cuda_unreachable_after_call
if errorlevel 1 goto tr_bootstrap_failed_after_call
goto after_torch_continue
:tr_cuda_unreachable_after_call
echo [train_registry] CUDA not usable: torch.cuda.is_available is False after cu124/cu121. See hints above ^(GPU/driver^?). Use --device cpu or TRAIN_REGISTRY_ALLOW_CPU=1.
exit /b 5
:tr_bootstrap_failed_after_call
echo [train_registry] pip / venv bootstrap failed - see stderr and steps above ^(omit TRAIN_REGISTRY_PIP_QUIET for verbose pip^).
exit /b 4
:after_torch_continue
"!EXE_PY!" "!TRPROBE!" short
if errorlevel 1 goto cuda_still_bad_after_verify
goto after_gpu_cuda_bootstrap

:cuda_still_bad_after_verify
if defined TRAIN_REGISTRY_ALLOW_CPU goto allow_cpu_after_cuda_fail
echo ERROR: CUDA still not available after bootstrap. Set TRAIN_REGISTRY_SKIP_CUDA_PIP=0 removed^? Try manual pip from pytorch.org
exit /b 3

:allow_cpu_after_cuda_fail
echo WARNING: continuing on CPU ^(TRAIN_REGISTRY_ALLOW_CPU=1^).

:after_gpu_cuda_bootstrap

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
REM %1 = python.exe path
REM Exit: 0 = CUDA OK, 1 = pip/ensurepip failure, 2 = CUDA false after cu124+cu121
:ensure_torch_cuda
set "_PY=%~1"
set "_PIPEXTRA="
if defined TRAIN_REGISTRY_PIP_QUIET set "_PIPEXTRA=-q"
echo [train_registry] Using: "!_PY!"
echo [train_registry] Step: ensure pip module
"!_PY!" -m pip --version >nul 2>&1
if errorlevel 1 goto ecc_run_ensurepip
goto ecc_after_ensurepip

:ecc_run_ensurepip
echo [train_registry] pip missing - running ensurepip...
"!_PY!" -m ensurepip --upgrade
if errorlevel 1 goto ecc_ensurepip_fail
goto ecc_after_ensurepip
:ecc_ensurepip_fail
echo [train_registry] ensurepip failed - reinstall Python with pip / use py -3.11 -m venv .venv
exit /b 1
:ecc_after_ensurepip

echo [train_registry] Step: pip install -U pip
"!_PY!" -m pip install %_PIPEXTRA% -U pip
if errorlevel 1 goto ecc_pip_up_fail
goto ecc_after_pip_up
:ecc_pip_up_fail
echo [train_registry] pip upgrade failed.
exit /b 1
:ecc_after_pip_up

echo [train_registry] Step: pip uninstall practice-htr ^(clean editable metadata^)
"!_PY!" -m pip uninstall -y practice-htr 2>nul

echo [train_registry] Step: pip install -e repo ^(dependencies^)
"!_PY!" -m pip install %_PIPEXTRA% -e "%REPO_ROOT%"
if errorlevel 1 goto ecc_editable_fail
goto ecc_after_editable
:ecc_editable_fail
echo [train_registry] pip install -e . failed offline or missing deps or bad Python version
exit /b 1
:ecc_after_editable

echo [train_registry] Step: probe torch CUDA
"!_PY!" "!TRPROBE!"
set "_TOR_PROBE=%ERRORLEVEL%"
if "!_TOR_PROBE!"=="0" exit /b 0

echo [train_registry] torch without working CUDA - reinstalling GPU wheels (cu124, then cu121). Needs internet.

"!_PY!" -m pip uninstall -y torch torchvision torchaudio 2>nul
echo [train_registry] Step: pip torch cu124
"!_PY!" -m pip install %_PIPEXTRA% --upgrade-strategy eager torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 goto ecc_cu124_warn
goto ecc_after_cu124_warn
:ecc_cu124_warn
echo [train_registry] cu124 wheel install failed network or URL
:ecc_after_cu124_warn

echo [train_registry] Step: probe after cu124
"!_PY!" "!TRPROBE!" short
set "_TOR_PROBE=%ERRORLEVEL%"
if "!_TOR_PROBE!"=="0" goto ecc_ok_cu124
goto ecc_after_cu124_probe
:ecc_ok_cu124
echo [train_registry] OK: CUDA after cu124
exit /b 0
:ecc_after_cu124_probe

echo [train_registry] Retrying CUDA wheels with cu121...
"!_PY!" -m pip uninstall -y torch torchvision torchaudio 2>nul
echo [train_registry] Step: pip torch cu121
"!_PY!" -m pip install %_PIPEXTRA% --upgrade-strategy eager torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 goto ecc_cu121_install_fail
goto ecc_after_cu121_install
:ecc_cu121_install_fail
echo [train_registry] cu121 wheel install failed.
exit /b 1
:ecc_after_cu121_install

echo [train_registry] Step: probe after cu121
"!_PY!" "!TRPROBE!" short
set "_TOR_PROBE=%ERRORLEVEL%"
if "!_TOR_PROBE!"=="0" goto ecc_ok_cu121
goto ecc_after_cu121_probe
:ecc_ok_cu121
echo [train_registry] OK: CUDA after cu121
exit /b 0
:ecc_after_cu121_probe

echo [train_registry] Still no CUDA from PyTorch on this interpreter.
echo [train_registry] nvidia-smi ^(driver / GPU in PATH^):
where nvidia-smi >nul 2>&1
if errorlevel 1 goto ecc_no_nvsmi
nvidia-smi -L
goto ecc_nvsmi_done
:ecc_no_nvsmi
echo [train_registry] nvidia-smi not found - no driver, VM without GPU passthrough, or wrong machine account.
:ecc_nvsmi_done
exit /b 2

:err
echo Train step failed, exit code %ERRORLEVEL%.
exit /b 1
