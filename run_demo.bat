@echo off
setlocal
cd /d "%~dp0"
py -3.11 -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
py -3.11 -m config_arbiter demo --ledger artifacts\demo-ledger.sqlite3 --report artifacts\acceptance_report.json
exit /b %errorlevel%

