@echo off
setlocal
python run_demo.py
set "QUANTCAL_EXIT=%ERRORLEVEL%"
echo Exit code: %QUANTCAL_EXIT%
exit /b %QUANTCAL_EXIT%
