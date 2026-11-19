@echo off
setlocal
cd /d "%~dp0"
python examples\run_demo.py
if errorlevel 1 exit /b %errorlevel%
echo.
echo Report: data\demo_report.json
endlocal
