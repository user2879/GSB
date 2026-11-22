@echo off
REM 可重复演示（Windows）：冷启动 -> 传播 -> 快照恢复 + 三类异常码
setlocal
cd /d "%~dp0.."
if exist artifacts\demo_run rmdir /s /q artifacts\demo_run
mkdir artifacts\demo_run
set WORK=artifacts\demo_run

echo == scenario 1 cold-start collection ==
python -m degprop --topology samples\topology.json --events samples\events.ndjson --ledger "%WORK%\session.db" --log-level WARNING
if errorlevel 1 exit /b 1

echo == scenario 3 snapshot resume and replay ==
python -m degprop --mode resume --topology samples\topology.json --events samples\events.ndjson --ledger "%WORK%\session.db" --log-level WARNING
if errorlevel 1 exit /b 1

echo == injection E_CYCLE ==
python -m degprop --inject cycle --ledger "%WORK%\cycle.db" --log-level WARNING
echo == injection E_CAPACITY ==
python -m degprop --inject capacity --ledger "%WORK%\cap.db" --log-level WARNING
echo == injection E_SNAPSHOT ==
python -m degprop --inject snapshot --ledger "%WORK%\snap.db" --log-level WARNING
echo == injection E_REPLAY ==
python -m degprop --inject replay --ledger "%WORK%\replay.db" --log-level WARNING
echo DEMO DONE
endlocal
