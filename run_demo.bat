@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -m imv.cli demo --data-dir "%~dp0artifacts\demo-workbench" %*
set EXIT_CODE=%ERRORLEVEL%
echo.
pause
exit /b %EXIT_CODE%
