@echo off
setlocal
cd /d "%~dp0\.."
if exist "workbench" rmdir /s /q "workbench"
if exist "artifacts\demo-workbench" rmdir /s /q "artifacts\demo-workbench"
echo Removed workbench\ and artifacts\demo-workbench\.
exit /b 0

