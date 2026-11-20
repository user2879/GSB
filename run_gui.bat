@echo off
setlocal
python --version
if errorlevel 1 goto :python_missing
python -c "import tkinter"
if errorlevel 1 goto :tk_missing
python -m sched_diag gui --nodes samples\nodes.json --pods samples\pods.json --plugins samples\plugins.json --workdir .sched-diag
exit /b %errorlevel%

:python_missing
echo Python was not found on PATH. Install Python 3.11 or later.
exit /b 1

:tk_missing
echo Tkinter is unavailable. Use the official Windows Python installer with Tcl/Tk selected.
exit /b 1
