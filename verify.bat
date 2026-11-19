@echo off
setlocal
cd /d "%~dp0"
python -m unittest discover -v
endlocal
