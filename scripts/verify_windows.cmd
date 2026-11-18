@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" call scripts\setup_windows.cmd
if errorlevel 1 exit /b 1
echo [1/3] Running pytest...
".venv\Scripts\python.exe" -m pytest -q
if errorlevel 1 exit /b 1
echo [2/3] Running three-scenario demo...
".venv\Scripts\python.exe" scripts\run_demo.py
if errorlevel 1 exit /b 1
echo [3/3] Running anomaly business-code checks...
".venv\Scripts\python.exe" scripts\run_anomaly_checks.py
exit /b %ERRORLEVEL%
