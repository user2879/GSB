@echo off
setlocal
cd /d "%~dp0"
python scripts\generate_demo.py
if errorlevel 1 exit /b %errorlevel%
python scripts\run_demo.py
if errorlevel 1 exit /b %errorlevel%
echo.
echo Demo complete. See artifacts\demo_summary.txt and artifacts\demo_report.json

