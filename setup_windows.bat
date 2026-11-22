@echo off
setlocal
cd /d "%~dp0"
py -3 --version >nul 2>&1
if errorlevel 1 (
  echo Python 3.10+ is required. Install Python and enable the py launcher.
  exit /b 1
)
py -3 -m pip install --upgrade pip
if errorlevel 1 exit /b 1
py -3 -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
py -3 scripts\validate_environment.py --workspace workbench
exit /b %errorlevel%
