@echo off
REM Аналог scripts/unzip_datasets.sh.
REM ZIP только через PowerShell Expand-Archive ^(не tar: иначе Git/cygwin tar в PATH даёт "Couldn't visit directory"^).
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0.."
set "ROOT=%CD%"
set "D=%ROOT%\data"

call :UnzipOne "%D%\digital_peter\images.zip" "%D%\digital_peter"
if errorlevel 1 exit /b 1
call :UnzipOne "%D%\yenisei_gov_reports_td\test_images.zip" "%D%\yenisei_gov_reports_td"
if errorlevel 1 exit /b 1
call :UnzipOne "%D%\yenisei_gov_reports_td\train_images.zip" "%D%\yenisei_gov_reports_td"
if errorlevel 1 exit /b 1
call :UnzipOne "%D%\russian_old_orthography_ocr\books-pdf-plaintext.zip" "%D%\russian_old_orthography_ocr"
if errorlevel 1 exit /b 1
call :UnzipOne "%D%\russian_old_orthography_ocr\pages-img-plaintext.zip" "%D%\russian_old_orthography_ocr"
if errorlevel 1 exit /b 1

if exist "%D%\digital_peter\__MACOSX" rd /s /q "%D%\digital_peter\__MACOSX"
echo Готово. Проверьте data\digital_peter\images и остальные каталоги.
exit /b 0

:UnzipOne
for %%I in ("%~1") do set "_Z=%%~fI"
for %%I in ("%~2\.") do set "_D=%%~fI"
if not exist "!_Z!" (
  echo Пропуск ^(нет файла^): !_Z!
  exit /b 0
)
echo Распаковка: !_Z! -^> !_D!
if not exist "!_D!" mkdir "!_D!"
set "EA_ZIP=!_Z!"
set "EA_DST=!_D!"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "& { $ErrorActionPreference='Stop'; $z=$Env:EA_ZIP; $d=$Env:EA_DST; if ([string]::IsNullOrWhiteSpace($z) -or [string]::IsNullOrWhiteSpace($d)) { exit 2 }; Expand-Archive -LiteralPath $z -DestinationPath $d -Force }"
set "PSX=!errorlevel!"
set "EA_ZIP="
set "EA_DST="
if not "!PSX!"=="0" (
  echo Ошибка Expand-Archive ^(powershell^), код !PSX! : !_Z!
  exit /b 1
)
exit /b 0
