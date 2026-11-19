@echo off
setlocal
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
python -m unittest discover -s tests -v
if errorlevel 1 pause
