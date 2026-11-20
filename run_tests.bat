@echo off
setlocal
python -m unittest discover -v
exit /b %errorlevel%
