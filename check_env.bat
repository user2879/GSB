@echo off
setlocal
cd /d "%~dp0"
py -3 scripts\check_windows_env.py --require-tk --open-window
if errorlevel 1 pause
