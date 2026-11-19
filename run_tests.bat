@echo off
setlocal
py -3.11 -m unittest discover -s tests -v
exit /b %errorlevel%
