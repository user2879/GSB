@echo off
REM Forced-termination drill: in-flight unknowns + byte-for-byte replay.
setlocal
cd /d "%~dp0.."
python scripts\crash_recovery_demo.py --rules samples\rules_v1.json
endlocal
