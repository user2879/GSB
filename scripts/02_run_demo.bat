@echo off
REM 确定性裁决 + sqlite 账本 + 快照，输出保留到 logs\demo_run.log
if not exist logs mkdir logs
python scripts\demo.py run > logs\demo_run.log 2>&1
type logs\demo_run.log
