@echo off
setlocal
cd /d "%~dp0"
python --version
python -m pip install -r requirements.txt
python demo.py %*
exit /b %ERRORLEVEL%
