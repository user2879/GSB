@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo [1/4] Checking Python...
python --version
if errorlevel 1 goto :python_missing
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)"
if errorlevel 1 goto :python_old
echo [2/4] Installing numpy...
python -m pip install -r requirements.txt
if errorlevel 1 goto :pip_failed
echo [3/4] Checking Tkinter...
python -c "import tkinter; print('Tk', tkinter.TkVersion)"
if errorlevel 1 goto :tk_missing
echo [4/4] Running acceptance suite...
python run.py demo --output artifacts
if errorlevel 1 goto :demo_failed
echo.
echo Setup complete. Launch with run.bat
exit /b 0

:python_missing
echo Python was not found. Install CPython 3.11+ from python.org and enable Add Python to PATH.
exit /b 1
:python_old
echo Python 3.11 or newer is required.
exit /b 1
:pip_failed
echo Failed to install dependencies.
exit /b 1
:tk_missing
echo Tkinter is unavailable. Re-run the official Windows Python installer and enable tcl/tk.
exit /b 1
:demo_failed
echo Acceptance suite failed. See artifacts\acceptance_report.json.
exit /b 1
