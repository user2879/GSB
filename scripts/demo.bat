@echo off
setlocal
cd /d "%~dp0\.."
py -3.11 -m mixedprec_diag.cli demo --config-dir configs --report-dir reports --snapshot-dir snapshots %*
exit /b %ERRORLEVEL%
