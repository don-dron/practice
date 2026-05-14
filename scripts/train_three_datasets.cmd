@echo off
setlocal
set ROOT=%~dp0..
set PYTHONPATH=%ROOT%\src;%PYTHONPATH%
htr-train --config "%ROOT%\configs\default.yaml" %*
