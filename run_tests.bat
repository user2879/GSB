@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
python -m pip install -r requirements-dev.txt
python -m pytest -q
