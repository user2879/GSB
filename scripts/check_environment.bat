@echo off
setlocal
cd /d "%~dp0\.."
python scripts\check_environment.py
exit /b %ERRORLEVEL%
