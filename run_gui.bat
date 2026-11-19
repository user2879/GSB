@echo off
cd /d "%~dp0"
py -3 run_gui.pyw
if errorlevel 1 pause
