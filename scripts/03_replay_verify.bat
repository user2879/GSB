@echo off
REM 依据快照恢复并一致性重放（E_REPLAY 校验）
python scripts\demo.py replay > logs\demo_replay.log 2>&1
type logs\demo_replay.log
