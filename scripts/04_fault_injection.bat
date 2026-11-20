@echo off
REM 三类异常注入：E_CHANNEL / E_ESTIMATE / E_SNAPSHOT
python scripts\demo.py faults > logs\demo_faults.log 2>&1
type logs\demo_faults.log
