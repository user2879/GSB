@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
python tools\generate_samples.py --seed 20261122 --count 12 --dim 8 --output data\samples.json
python -m unittest discover -s tests -v
if errorlevel 1 exit /b %errorlevel%
python -m memdiag.cli check-incremental --plan data\plan.json --samples data\samples.json --changes data\changes_scale.json --report artifacts\incremental_check.json
if errorlevel 1 exit /b %errorlevel%
endlocal & exit /b 0
