@echo off
REM Deterministic six-scenario demonstration (Windows, Python 3.11/3.12, stdlib only).
setlocal
cd /d "%~dp0.."
if not exist artifacts mkdir artifacts
python -m snapshot_migration.cli demo --chain samples\snapshot_chain.json --out artifacts\demo_output.txt
echo.
echo Demo output written to artifacts\demo_output.txt
endlocal
