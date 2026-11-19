@echo off
setlocal
cd /d "%~dp0\.."
python -m cdn_edge verify %*
exit /b %ERRORLEVEL%
