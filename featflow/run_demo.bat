@echo off
REM Deterministic featflow demo for Windows command line.
setlocal
cd /d %~dp0
python scripts\generate_inputs.py --tier small --out data
if errorlevel 1 exit /b 1
python run_pipeline.py --data data --out out\small --serial
if errorlevel 1 exit /b 1
python run_pipeline.py --data data --out out\small --mode resume
echo Demo finished. See out\small\run_report.json
