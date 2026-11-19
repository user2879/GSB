@echo off
setlocal
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
python demo.py run --output artifacts\run_report.json
if errorlevel 1 pause
