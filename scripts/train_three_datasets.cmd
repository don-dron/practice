@echo off
REM Historical name: default.yaml trains on Yenisei Gov Reports TD only.
setlocal
set ROOT=%~dp0..
set PYTHONPATH=%ROOT%\src;%PYTHONPATH%
htr-train --config "%ROOT%\configs\default.yaml" %*
