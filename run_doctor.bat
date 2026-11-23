@echo off
setlocal
cd /d "%~dp0"
py -3 -m gray_merge_lab doctor --spec
if errorlevel 1 pause

