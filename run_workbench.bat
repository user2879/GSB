@echo off
setlocal
cd /d "%~dp0"
python -m adw --config config.yaml %*
exit /b %ERRORLEVEL%
