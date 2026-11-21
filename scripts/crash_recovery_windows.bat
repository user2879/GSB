@echo off
setlocal
set ROOT=%~dp0..
set WORK=%ROOT%\work\crash
if not exist %WORK%\samples mkdir %WORK%\samples
pushd %ROOT%
python -m emergency_drill.cli generate-samples --sample-dir %WORK%\samples
python -m emergency_drill.cli crash-demo --plan %WORK%\samples\plan_valid.json --events %WORK%\samples\events_valid.json --ledger %WORK%\crash.sqlite3 --crash-stage detect
python -m emergency_drill.cli --log %WORK%\crash.log.jsonl recover --ledger %WORK%\crash.sqlite3 --output %WORK%\recover.json
echo recover returned %ERRORLEVEL% (expected 2)
python -m emergency_drill.cli rerun-after-crash --ledger %WORK%\crash.sqlite3 --output %WORK%\rerun.json
popd
endlocal
