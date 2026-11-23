@echo off
setlocal
cd /d "%~dp0"
python -m config_diag gui
exit /b %errorlevel%
