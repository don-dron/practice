@echo off
REM Windows: аналог scripts/archive_data_datasets.sh (pack | fetch).
REM Требуется curl.exe и %SystemRoot%\System32\tar.exe (не tar из Git ^— иначе часто ломается^), powershell.exe
REM только одной строкой — запрос JSON с Яндекса (.ps1-файл не нужен).
REM
REM   scripts\archive_data_datasets.cmd fetch
REM   scripts\archive_data_datasets.cmd pack

setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0\.."
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
set COPYFILE_DISABLE=1
echo Packing to practice_data_datasets.tar.gz (large file, wait...)
set "WTAR=%SystemRoot%\System32\tar.exe"
if not exist "!WTAR!" set "WTAR=tar"
set "ARC=%ROOT%\practice_data_datasets.tar.gz"
REM bsdtar: -C задаёт базу имён как в bash; без этого и без cd бывают пустые члены архива под Windows.
if not exist "!ROOT!\data\yenisei_gov_reports_td" (
  echo ERROR: folder missing: "!ROOT!\data\yenisei_gov_reports_td" - run scripts\archive_data_datasets.cmd fetch first, then pack.
  exit /b 1
)
"!WTAR!" -czf "!ARC!" -C "!ROOT!" "data\yenisei_gov_reports_td"
set "TA=!errorlevel!"
if not "!TA!"=="0" (
  echo Tar pack failed ^(exit !TA!^).
  exit /b 1
)
dir "!ARC!"
echo Done ^(pack^).
exit /b 0

:: ---------------------------------------------------------------------------
:Fetch
set "DOWN=%ROOT%\archive.tar.gz"

echo Asking Yandex.Disk API for download link...
REM Same as .sh: curl JSON to file.
set "JS=%TEMP%\practice_yndx_meta_%RANDOM%.json"
curl.exe -fsS "%META_URL%" -o "!JS!"
if errorlevel 1 (
  echo ERROR: curl metadata request failed.
  del "!JS!" 2>nul
  exit /b 1
)
set "META_JSON=!JS!"
set "HREF="
for /f "delims=" %%U in ('powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $p = $Env:META_JSON; if ([string]::IsNullOrWhiteSpace($p)) { exit 11 }; $t = Get-Content -LiteralPath $p -Raw -Encoding utf8; $o = ConvertFrom-Json $t -ErrorAction Stop; $h = $o.href; if ($null -eq $h) { exit 11 }; if (($h -is [string]) -and [string]::IsNullOrWhiteSpace($h)) { exit 11 }; Write-Output $h }"') do set "HREF=%%U"
if "!HREF!"=="" (
  echo ERROR: no href in API response. First bytes of API body:
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& { $p = $Env:META_JSON; if (![string]::IsNullOrWhiteSpace($p) -and (Test-Path -LiteralPath $p)) { $t = Get-Content -LiteralPath $p -Raw -Encoding utf8; if ($t.Length -gt 350) { $t = $t.Substring(0,350) }; $t } }" 2>nul
  set "META_JSON="
  del "!JS!" 2>nul
  exit /b 1
)
set "META_JSON="
del "!JS!" 2>nul

echo Downloading archive...
curl.exe -fsSL "!HREF!" -o "!DOWN!"
if errorlevel 1 (
  echo ERROR: curl download failed.
  exit /b 1
)

set "TMPD=%TEMP%\practice_data_unpack_%RANDOM%"
mkdir "!TMPD!" 2>nul
if errorlevel 1 (
  echo ERROR: cannot create temp folder "!TMPD!"
  exit /b 1
)

echo Extracting tarball to temp...
set "WTAR=%SystemRoot%\System32\tar.exe"
if not exist "!WTAR!" set "WTAR=tar"
set "_BACK=%CD%"
cd /d "!TMPD!"
if errorlevel 1 (
  echo ERROR: cannot cd to "!TMPD!"
  exit /b 1
)
"!WTAR!" -xzf "!DOWN!"
set "_T=!errorlevel!"
cd /d "!_BACK!"
if not "!_T!"=="0" (
  rd /s /q "!TMPD!" 2>nul
  echo ERROR: tar extract failed.
  exit /b 1
)

REM yenisei_gov_reports_td at tarball root OR inside one wrapper folder (older bundles may ship extra dirs).
set "TMP_UNPACK=!TMPD!"
set "DATAROOT="
for /f "delims=" %%B in ('powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& { $tmp = $Env:TMP_UNPACK; if (-not $tmp) { exit 2 }; $y = Join-Path $tmp 'yenisei_gov_reports_td'; if (Test-Path -LiteralPath $y -PathType Container) { [Console]::WriteLine($tmp); exit 0 }; foreach ($di in @(Get-ChildItem -LiteralPath $tmp -Directory -ErrorAction SilentlyContinue)) { $yy = Join-Path $di.FullName 'yenisei_gov_reports_td'; if (Test-Path -LiteralPath $yy -PathType Container) { [Console]::WriteLine($di.FullName); exit 0 } }; exit 3 }"') do set "DATAROOT=%%B"
set "TMP_UNPACK="
if not defined DATAROOT (
  echo ERROR: tarball has no yenisei_gov_reports_td folder at root or under a single subdirectory
  echo Listing top level of temp:
  dir /b /ad "!TMPD!"
  rd /s /q "!TMPD!" 2>nul
  exit /b 1
)

mkdir "%ROOT%\data" 2>nul

for %%D in (yenisei_gov_reports_td) do (
  if not exist "!DATAROOT!\%%D" (
    echo ERROR: missing folder !DATAROOT!\%%D after layout detect
    rd /s /q "!TMPD!" 2>nul
    exit /b 1
  )
  if exist "%ROOT%\data\%%D" rd /s /q "%ROOT%\data\%%D"
  mkdir "%ROOT%\data" 2>nul
  robocopy "!DATAROOT!\%%D" "%ROOT%\data\%%D" /E /MOVE /R:2 /W:2 /NFL /NDL /NJH /NJS /nc /ns /np
  set "RO=!errorlevel!"
  if !RO! GEQ 8 (
    echo ERROR: robocopy failed for %%D code=!RO!
    rd /s /q "!TMPD!" 2>nul
    exit /b 1
  )
)

rd /s /q "!TMPD!" 2>nul

dir "!DOWN!"
echo Done ^(fetch^). Run: scripts\unzip_datasets.cmd
exit /b 0

REM ---------------------------------------------------------------------------
:UsageFail
exit /b 2

:Usage
echo Использование: %~nx0 [pack ^| fetch ^| download]
echo   fetch/download — скачать archive.tar.gz и папку data\yenisei_gov_reports_td (в архиве допускаются лишние каталоги)
echo   pack           — упаковать data\ в practice_data_datasets.tar.gz ^(тяжёлое^)
exit /b 2
