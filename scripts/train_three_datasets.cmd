@echo off
REM Historical name: default.yaml merges two COCO corpuses (Yenisei + Digital Peter).
setlocal
set ROOT=%~dp0..
set PYTHONPATH=%ROOT%\src;%PYTHONPATH%
htr-train --config "%ROOT%\configs\default.yaml" %*
