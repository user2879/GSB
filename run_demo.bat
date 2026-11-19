@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
py -3.11 -m story_workbench demo --out artifacts\demo
if errorlevel 1 exit /b %errorlevel%
echo.
echo Demo report: artifacts\demo\report.md
