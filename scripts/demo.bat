@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0\.."
python -m pip install -r requirements.txt
python samples\generate_samples.py --seed 20261122
python scripts\demo.py
exit /b %ERRORLEVEL%
