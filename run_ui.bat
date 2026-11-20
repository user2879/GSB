@echo off
setlocal
cd /d "%~dp0"
py -3 -c "import tkinter, numpy, sqlite3" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] tkinter/numpy/sqlite3 unavailable. Enable tcl/tk in the Python installer.
  exit /b 2
)
py -3 -m workbench.ui %*
exit /b %errorlevel%
