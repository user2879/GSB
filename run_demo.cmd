@echo off
setlocal
cd /d "%~dp0"
python run_demo.py --workspace "%~dp0demo_workspace" %*
exit /b %ERRORLEVEL%
