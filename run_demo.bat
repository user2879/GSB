@echo off
REM Deterministic ML pipeline diagnostic workbench - Windows demo entry point.
REM Requires Python 3.10+ on PATH with tcl/tk installed.
setlocal
python --version
if errorlevel 1 (
  echo Python was not found on PATH. Install Python 3.10+ and retry.
  exit /b 1
)
python -m workbench.cli check-env
python scripts\demo.py
exit /b %errorlevel%
