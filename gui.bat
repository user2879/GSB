@echo off
setlocal
cd /d "%~dp0"
python -m batch_audit.cli gui --workspace workspace
exit /b %ERRORLEVEL%
