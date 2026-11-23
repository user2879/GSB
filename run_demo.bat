@echo off
setlocal
cd /d "%~dp0"
py -3 -m gray_merge_lab demo --out artifacts\demo
if errorlevel 1 pause

