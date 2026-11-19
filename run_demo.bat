@echo off
REM Deterministic RTS replay workbench - fixed demo sequence (Windows)
setlocal
cd /d "%~dp0"
if not exist workspace mkdir workspace
python -m rts_replay --workspace workspace demo
if errorlevel 1 exit /b %errorlevel%
python -m unittest discover -s tests
endlocal
