@echo off
setlocal
python "%~dp0run.py" demo %*
exit /b %ERRORLEVEL%
