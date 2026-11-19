@echo off
REM Windows entry point for the deterministic demonstration.
REM Requires Python 3.11+ on PATH and `pip install numpy`.
python scripts\demo.py
exit /b %ERRORLEVEL%
