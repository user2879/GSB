@echo off
setlocal
cd /d "%~dp0.."
py -3.11 -m snapdiag doctor --workspace .
if errorlevel 1 exit /b %errorlevel%
py -3.11 -m snapdiag demo --workspace .
exit /b %errorlevel%

