@echo off
setlocal
cd /d "%~dp0"
python -m shardrep demo --root . %*
if errorlevel 1 exit /b %errorlevel%
echo.
echo Demo report: demo-run\demo_report.json
echo Structured log: demo-run\logs\events.jsonl
echo SQLite ledger: demo-run\ledger.sqlite3

