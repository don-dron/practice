@echo off
REM Аналог scripts/unzip_datasets.sh — только cmd + tar.exe (ZIP тоже умеет).
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0.."
set "ROOT=%CD%"
set "D=%ROOT%\data"

call :UnzipQuiet "%D%\digital_peter\images.zip" "%D%\digital_peter"
if errorlevel 1 exit /b 1
call :UnzipQuiet "%D%\yenisei_gov_reports_td\test_images.zip" "%D%\yenisei_gov_reports_td"
if errorlevel 1 exit /b 1
call :UnzipQuiet "%D%\yenisei_gov_reports_td\train_images.zip" "%D%\yenisei_gov_reports_td"
if errorlevel 1 exit /b 1
call :UnzipQuiet "%D%\russian_old_orthography_ocr\books-pdf-plaintext.zip" "%D%\russian_old_orthography_ocr"
if errorlevel 1 exit /b 1
call :UnzipQuiet "%D%\russian_old_orthography_ocr\pages-img-plaintext.zip" "%D%\russian_old_orthography_ocr"
if errorlevel 1 exit /b 1

if exist "%D%\digital_peter\__MACOSX" rd /s /q "%D%\digital_peter\__MACOSX"
echo Готово. Проверьте data\digital_peter\images и остальные каталоги.
exit /b 0

:UnzipQuiet
set "ZIP=%~1"
set "DEST=%~2"
if not exist "!ZIP!" (
  echo Пропуск ^(нет файла^): !ZIP!
  exit /b 0
)
echo Распаковка: !ZIP! -^> !DEST!
mkdir "%DEST%" 2>nul
REM Встроенный tar иногда не тянет отдельные ZIP ^(Deflate64, AES и т.п.^): тогда Expand-Archive.
set "UNZIP_LITERAL_ZIP=!ZIP!"
set "UNZIP_LITERAL_DST=!DEST!"
tar.exe -xf "!ZIP!" -C "!DEST!"
if errorlevel 1 (
  echo tar не смог ZIP — пробуем Expand-Archive ^(как unzip под Linux^)...
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& { $ErrorActionPreference = 'Stop'; $z = $Env:UNZIP_LITERAL_ZIP; $d = $Env:UNZIP_LITERAL_DST; Expand-Archive -LiteralPath $z -DestinationPath $d -Force }"
  if errorlevel 1 (
    echo Ошибка распаковки ZIP ^(tar и Expand-Archive^): !ZIP!
    set "UNZIP_LITERAL_ZIP="
    set "UNZIP_LITERAL_DST="
    exit /b 1
  )
)
set "UNZIP_LITERAL_ZIP="
set "UNZIP_LITERAL_DST="
exit /b 0
