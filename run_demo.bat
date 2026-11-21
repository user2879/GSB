@echo off
setlocal
cd /d "%~dp0"
py -3 -m mlpipe_workbench demo --out-dir demo_output
if errorlevel 1 pause
