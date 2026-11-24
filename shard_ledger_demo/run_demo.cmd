@echo off
REM Repeatable local demo for the cross-node shard quota ledger (Windows)
setlocal
set WD=%~dp0runs\demo
if not exist "%WD%" mkdir "%WD%"
python -m shard_ledger.demo run --manifest samples\manifest.json --events samples\events.ndjson --workdir "%WD%"
python -m shard_ledger.demo verify --manifest samples\manifest.json --events samples\events.ndjson --workdir "%WD%"
endlocal
