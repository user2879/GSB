@echo off
setlocal
cd /d "%~dp0"
py -3 -c "import numpy, sqlite3" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python 3 with numpy and sqlite3 is required.
  exit /b 2
)
py -3 demo.py %*
exit /b %errorlevel%
