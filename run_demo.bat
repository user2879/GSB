@echo off
setlocal
python --version
if errorlevel 1 exit /b 1
python -m sched_diag demo --workdir artifacts\demo-state
exit /b %errorlevel%
