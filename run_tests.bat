@echo off
setlocal
cd /d "%~dp0"
python -m unittest -v tests.test_workbench
exit /b %ERRORLEVEL%
