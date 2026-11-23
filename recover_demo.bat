@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

python -m eventbus_diag recover --spec samples\spec.json --events samples\events.ndjson --db artifacts\ledger.sqlite --log artifacts\bus.jsonl
endlocal
