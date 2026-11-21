@echo off
REM 应急预案演练诊断工作台 - Windows 本地验收脚本（无需管理员、无网络、无容器）
setlocal
set ROOT=%~dp0..
set WORK=%ROOT%\work\acceptance
set PY=python
pushd %ROOT%
if not exist %WORK%\samples mkdir %WORK%\samples
%PY% -m emergency_drill.cli --log %WORK%\drill.log.jsonl generate-samples --sample-dir %WORK%\samples
if errorlevel 1 goto fail

%PY% -m emergency_drill.cli --log %WORK%\drill.log.jsonl run --plan %WORK%\samples\plan_valid.json --events %WORK%\samples\events_valid.json --ledger %WORK%\drill.sqlite3 --output %WORK%\run.json
if errorlevel 1 goto fail

(
echo {"operations":[
echo   {"op":"update_duration","stage_id":"recover","duration_ms":1700},
echo   {"op":"rewrite_criterion","stage_id":"recover","criterion":{"mode":"all_of","required_actions":["a_service_restore"]}}
echo ]}
) > %WORK%\changeset.json
%PY% -m emergency_drill.cli --log %WORK%\drill.log.jsonl edit --ledger %WORK%\drill.sqlite3 --changeset %WORK%\changeset.json --output %WORK%\edit.json
if errorlevel 1 goto fail
%PY% -m emergency_drill.cli --log %WORK%\drill.log.jsonl verify-incr --ledger %WORK%\drill.sqlite3
if errorlevel 1 goto fail
%PY% -m emergency_drill.cli --log %WORK%\drill.log.jsonl snapshot --ledger %WORK%\drill.sqlite3 --output %WORK%\snapshot.json
%PY% -m emergency_drill.cli --log %WORK%\drill.log.jsonl replay --ledger %WORK%\drill.sqlite3 --snapshot %WORK%\snapshot.json
if errorlevel 1 goto fail
%PY% -m emergency_drill.cli --log %WORK%\drill.log.jsonl tamper-snapshot --snapshot %WORK%\snapshot.json --output %WORK%\snapshot.bad.json
%PY% -m emergency_drill.cli --log %WORK%\drill.log.jsonl open-snapshot --ledger %WORK%\drill.sqlite3 --snapshot %WORK%\snapshot.bad.json
%PY% -m emergency_drill.cli --log %WORK%\drill.log.jsonl view --ledger %WORK%\drill.sqlite3 --text
goto end
:fail
echo ACCEPTANCE FAILED
exit /b 1
:end
popd
endlocal
