@echo off
setlocal
cd /d "%~dp0"
python -m meshdiag ui --workspace samples\workspace.json
exit /b %ERRORLEVEL%
