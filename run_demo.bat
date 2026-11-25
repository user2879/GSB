@echo off
setlocal
cd /d "%~dp0"
python --version
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
python run_demo.py
if errorlevel 1 exit /b 1
python -m unittest discover -s tests
exit /b %errorlevel%
