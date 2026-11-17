@echo off
setlocal
cd /d "%~dp0"
python -m exp_tracker gui --workspace demo_workspace
if errorlevel 1 exit /b %errorlevel%
endlocal

