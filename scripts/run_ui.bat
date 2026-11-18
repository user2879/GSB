@echo off
setlocal
cd /d "%~dp0\.."
py -3.11 -m mixedprec_diag.cli ui --config configs/float32.json --auto-run %*
exit /b %ERRORLEVEL%
