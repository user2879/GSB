@echo off
setlocal
cd /d "%~dp0"
py -3 -m slowdiag demo --workspace workbench --pretty
exit /b %errorlevel%
