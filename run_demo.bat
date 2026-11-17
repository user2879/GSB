@echo off
setlocal
cd /d "%~dp0"
python demo.py artifacts\demo
exit /b %ERRORLEVEL%

