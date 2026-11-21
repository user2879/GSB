@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0\.."
python -m pip install -r requirements.txt
python -m unittest discover -v
exit /b %ERRORLEVEL%
