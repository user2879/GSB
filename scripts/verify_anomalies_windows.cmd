@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" call scripts\setup_windows.cmd
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" scripts\run_anomaly_checks.py
exit /b %ERRORLEVEL%
