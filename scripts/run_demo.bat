@echo off
setlocal
cd /d "%~dp0.."
if not exist .venv\Scripts\python.exe call scripts\setup_windows.bat
.venv\Scripts\python.exe run_demo.py
endlocal
