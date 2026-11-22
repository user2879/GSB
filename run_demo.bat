@echo off
setlocal
python app.py demo --workspace demo_workspace --reset
if errorlevel 1 exit /b %errorlevel%
python -m pytest -q
if errorlevel 1 exit /b %errorlevel%
python verify_acceptance.py
