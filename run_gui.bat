@echo off
setlocal
cd /d "%~dp0"
py -3.10 -m walcrash --workspace demo_workspace --gui
exit /b %ERRORLEVEL%
