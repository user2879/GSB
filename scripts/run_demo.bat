@echo off
REM Deterministic three-round demo on Windows (no third-party packages).
cd /d "%~dp0\.."
python -m pool_workbench.demo
pause
