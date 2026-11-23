@echo off
setlocal
cd /d "%~dp0"
python -m unittest discover -s tests -v
set CODE=%ERRORLEVEL%
echo.
echo Test exit code: %CODE%
pause
exit /b %CODE%
