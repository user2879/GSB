@echo off
setlocal
cd /d "%~dp0"
python scripts\demo.py %*
if errorlevel 1 exit /b %errorlevel%
endlocal

