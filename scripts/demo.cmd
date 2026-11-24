@echo off
setlocal
cd /d "%~dp0\.."
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
python -m mmledger demo --root . --jobs 4
if errorlevel 1 exit /b 1
python scripts\verify.py
