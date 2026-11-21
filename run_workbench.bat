@echo off
setlocal
cd /d "%~dp0"
py -3 -m mlpipe_workbench ui --db workbench.db --data data\sample.json --pipeline data\pipeline.json
if errorlevel 1 pause
