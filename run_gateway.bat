@echo off
setlocal
cd /d "%~dp0"
py -3 -m gray_merge_lab serve --gateway-port 28190
if errorlevel 1 pause

