@echo off
setlocal
cd /d "%~dp0"
python demo.py
exit /b %ERRORLEVEL%
