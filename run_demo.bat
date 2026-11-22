@echo off
setlocal
python -m unittest discover -v
if errorlevel 1 exit /b 1
python -m configdist demo --out artifacts\demo\acceptance.txt
