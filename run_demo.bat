@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
python -m joinlab demo --workload samples\demo_workload.json --db artifacts\demo_ledger.sqlite --out artifacts\demo_report.json --reset %*
if errorlevel 1 exit /b %errorlevel%
echo.
echo Demo report: artifacts\demo_report.json
echo Ledger:      artifacts\demo_ledger.sqlite
