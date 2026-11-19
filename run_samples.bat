@echo off
setlocal
cd /d "%~dp0"
py -3 -m pip install -r requirements.txt
if errorlevel 1 exit /b %errorlevel%
py -3 -m rts_pipeline sample --out samples
if errorlevel 1 exit /b %errorlevel%
py -3 -m unittest discover -v
exit /b %errorlevel%
