@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist demo_output mkdir demo_output
py -3 app.py --ui --spec samples\spec.json --events samples\normal.ndjson --db demo_output\ui.sqlite3
exit /b %errorlevel%
