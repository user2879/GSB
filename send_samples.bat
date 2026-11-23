@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  set PORT=28190
) else (
  set PORT=%~1
)
if "%~2"=="" (
  set SAMPLES=gray_merge_lab\data\samples.ndjson
) else (
  set SAMPLES=%~2
)
py -3 scripts\send_samples.py --port %PORT% "%SAMPLES%"
if errorlevel 1 pause

