@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
py -3 demo.py
exit /b %errorlevel%
