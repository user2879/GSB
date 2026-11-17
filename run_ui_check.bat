@echo off
setlocal
cd /d "%~dp0"
python -m autoscale_diag ui --db artifacts\demo\workbench.db --run-id demo --check
exit /b %ERRORLEVEL%

