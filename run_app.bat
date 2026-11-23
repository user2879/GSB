@echo off
setlocal
cd /d "%~dp0"
py -3 -m gray_merge_lab doctor --spec
py -3 -m gray_merge_lab gui --workdir artifacts\app
if errorlevel 1 pause

