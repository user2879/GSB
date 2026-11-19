@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
python -m volumetric_editor demo --root .
exit /b %ERRORLEVEL%
