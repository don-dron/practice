@echo off
REM Windows: аналог scripts/archive_data_datasets.sh (pack | fetch).
REM Требуется curl.exe и tar.exe (Windows 10+), для fetch — доступный powershell.exe
REM только одной строкой — запрос JSON с Яндекса (.ps1-файл не нужен).
REM
REM   scripts\archive_data_datasets.cmd fetch
REM   scripts\archive_data_datasets.cmd pack

setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0.."
set "ROOT=%CD%"
set "ACTION=%~1"
if "%ACTION%"=="" set "ACTION=pack"

set "META_URL=https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=https://disk.yandex.ru/d/dXi_n8w5Q-U5iA"

if /i "%ACTION%"=="-h" goto Usage
if /i "%ACTION%"=="--help" goto Usage
if /i "%ACTION%"=="help" goto Usage

if /i "%ACTION%"=="pack" goto Pack
if /i "%ACTION%"=="fetch" goto Fetch
if /i "%ACTION%"=="download" goto Fetch

echo Неизвестная команда: %ACTION%
goto UsageFail

:: ---------------------------------------------------------------------------
:Pack
pushd "%ROOT%"
set COPYFILE_DISABLE=1
echo Создание: practice_data_datasets.tar.gz (может быть несколько ГБ^)...
tar -czf "%ROOT%\practice_data_datasets.tar.gz" ^
  "data/digital_peter" ^
  "data/yenisei_gov_reports_td" ^
  "data/russian_old_orthography_ocr"
popd
if errorlevel 1 (
  echo Ошибка tar ^(pack^).
  exit /b 1
)
dir "%ROOT%\practice_data_datasets.tar.gz"
echo Готово ^(pack^).
exit /b 0

:: ---------------------------------------------------------------------------
:Fetch
set "DOWN=%ROOT%\archive.tar.gz"

echo Получение прямой ссылки с Яндекс.Диска...
REM Как в archive_data_datasets.sh: метаданные — через curl (.json во временный файл).
REM Invoke-RestMethod с public_key=https://… ломался на некоторых системах URL-парсером.
set "JS=%TEMP%\practice_yndx_meta_%RANDOM%.json"
curl.exe -fsS "%META_URL%" -o "!JS!"
if errorlevel 1 (
  echo Ошибка: запрос метаданных к API Яндекс.Диска ^(curl^).
  del "!JS!" 2>nul
  exit /b 1
)
set "META_JSON=!JS!"
set "HREF="
for /f "delims=" %%U in ('powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $p = $Env:META_JSON; if ([string]::IsNullOrWhiteSpace($p)) { exit 11 }; $t = Get-Content -LiteralPath $p -Raw -Encoding utf8; $o = ConvertFrom-Json $t -ErrorAction Stop; $h = $o.href; if ($null -eq $h) { exit 11 }; if (($h -is [string]) -and [string]::IsNullOrWhiteSpace($h)) { exit 11 }; Write-Output $h }"') do set "HREF=%%U"
if "!HREF!"=="" (
  echo Ошибка: в ответе API нет href. Первые символы ответа:
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& { $p = $Env:META_JSON; if (![string]::IsNullOrWhiteSpace($p) -and (Test-Path -LiteralPath $p)) { $t = Get-Content -LiteralPath $p -Raw -Encoding utf8; if ($t.Length -gt 350) { $t = $t.Substring(0,350) }; $t } }" 2>nul
  echo ^(сообщите преподавателю, если здесь виден HTML/XML вместо JSON.^)
  set "META_JSON="
  del "!JS!" 2>nul
  exit /b 1
)
set "META_JSON="
del "!JS!" 2>nul

echo Загрузка архива...
curl.exe -fsSL "!HREF!" -o "!DOWN!"
if errorlevel 1 (
  echo Ошибка curl при скачивании архива.
  exit /b 1
)

set "TMPD=%TEMP%\practice_data_unpack_%RANDOM%"
mkdir "!TMPD!" 2>nul
if errorlevel 1 (
  echo Не удалось создать временную папку.
  exit /b 1
)

echo Распаковка архива во временную папку...
pushd "!TMPD!"
tar.exe -xzf "!DOWN!"
set "_T=!errorlevel!"
popd
if not "!_T!"=="0" (
  rd /s /q "!TMPD!" 2>nul
  echo Ошибка tar при распаковке.
  exit /b 1
)

mkdir "%ROOT%\data" 2>nul

for %%D in (digital_peter russian_old_orthography_ocr yenisei_gov_reports_td) do (
  if not exist "!TMPD!\%%D\" (
    echo Ошибка: в архиве нет каталога %%D в корне.
    rd /s /q "!TMPD!" 2>nul
    exit /b 1
  )
  if exist "%ROOT%\data\%%D" rd /s /q "%ROOT%\data\%%D"
  mkdir "%ROOT%\data" 2>nul
  REM move между дисками ^(например TEMP на C:, репо на D:^) падает — robocopy /MOVE работает везде.
  robocopy "!TMPD!\%%D" "%ROOT%\data\%%D" /E /MOVE /R:2 /W:2 /NFL /NDL /NJH /NJS /nc /ns /np
  set "RO=!errorlevel!"
  if !RO! GEQ 8 (
    echo Ошибка robocopy при переносе %%D в data\ ^(код !RO!, см. сообщения robocopy^).
    rd /s /q "!TMPD!" 2>nul
    exit /b 1
  )
)

rd /s /q "!TMPD!" 2>nul

dir "!DOWN!"
echo Готово ^(fetch^): см. каталог data\ digital_peter, russian_old_orthography_ocr, yenisei_gov_reports_td
echo Затем: scripts\unzip_datasets.cmd
exit /b 0

REM ---------------------------------------------------------------------------
:UsageFail
exit /b 2

:Usage
echo Использование: %~nx0 [pack ^| fetch ^| download]
echo   fetch/download — скачать archive.tar.gz и три папки в data\digital_peter и др.
echo   pack           — упаковать data\ в practice_data_datasets.tar.gz ^(тяжёлое^)
exit /b 2
