@echo off
setlocal
cd /d "%~dp0\.."
python --version >nul 2>&1
if errorlevel 1 (
  echo Python is not available on PATH as "python". Install Python 3.11+ first.
  exit /b 1
)
if not exist workspaces mkdir workspaces
python scripts\run_demo.py "%CD%\workspaces\demo-run"
endlocal
