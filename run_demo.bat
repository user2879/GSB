@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

python -m eventbus_diag validate --spec samples\spec.json --events samples\events.ndjson
if errorlevel 1 exit /b %errorlevel%

python -m eventbus_diag run --spec samples\spec.json --events samples\events.ndjson --db artifacts\ledger.sqlite --log artifacts\bus.jsonl --fresh
if errorlevel 1 exit /b %errorlevel%

python -m eventbus_diag ui --spec samples\spec.json --events samples\events.ndjson --db artifacts\ledger.sqlite --log artifacts\bus.jsonl
endlocal
