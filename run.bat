@echo off
setlocal
cd /d "%~dp0"
python -m batch_audit.cli run --workspace workspace %*
exit /b %ERRORLEVEL%
