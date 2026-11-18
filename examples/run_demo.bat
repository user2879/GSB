@echo off
REM Windows entry point for the deterministic demonstration.
REM Requires Python 3.10+ on PATH and numpy installed (pip install -r requirements.txt).
setlocal
cd /d "%~dp0\.."
python examples\run_demo.py
endlocal
