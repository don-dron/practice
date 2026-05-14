@echo off
REM Like scripts/unzip_datasets.sh — extracts inner .zip via PowerShell Expand-Archive.
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0\.."
set "ROOT=%CD%"
set "D=%ROOT%\data"

if not exist "!D!" (
  echo ERROR: folder does not exist: "!D!"
  echo Download datasets first:
  echo   .\scripts\archive_data_datasets.cmd fetch
  echo Then run:
  echo   .\scripts\unzip_datasets.cmd
  exit /b 1
)

set ZCOUNT=0
call :UnzipOne "%D%\yenisei_gov_reports_td\test_images.zip" "%D%\yenisei_gov_reports_td"
if errorlevel 1 exit /b 1
call :UnzipOne "%D%\yenisei_gov_reports_td\train_images.zip" "%D%\yenisei_gov_reports_td"
if errorlevel 1 exit /b 1

if "!ZCOUNT!"=="0" (
  echo ERROR: none of the expected zip files were found under:
  echo   "!D!"
  echo You only have unpacked folders — or fetch did not finish. Run:
  echo   .\scripts\archive_data_datasets.cmd fetch
  echo Expected files include: "!D!\yenisei_gov_reports_td\train_images.zip" ...
  exit /b 1
)
if !ZCOUNT! LSS 2 (
  echo WARNING: found only !ZCOUNT! of 2 zip archives ok; some datasets may still be unpacked.
)

echo Done. Check data\yenisei_gov_reports_td\train_images and test_images
exit /b 0

:UnzipOne
for %%I in ("%~1") do set "_Z=%%~fI"
for %%I in ("%~2\.") do set "_D=%%~fI"
if not exist "!_Z!" (
  echo SKIP missing: "!_Z!"
  exit /b 0
)
set /a ZCOUNT+=1
echo Extract: "!_Z!" -^> "!_D!"
if not exist "!_D!" mkdir "!_D!"
set "EA_ZIP=!_Z!"
set "EA_DST=!_D!"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "& { $ErrorActionPreference='Stop'; $z=$Env:EA_ZIP; $d=$Env:EA_DST; if ([string]::IsNullOrWhiteSpace($z) -or [string]::IsNullOrWhiteSpace($d)) { exit 2 }; Expand-Archive -LiteralPath $z -DestinationPath $d -Force }"
set "PSX=!errorlevel!"
set "EA_ZIP="
set "EA_DST="
if not "!PSX!"=="0" (
  echo ERROR Expand-Archive exit=!PSX! file="!_Z!"
  exit /b 1
)
exit /b 0
