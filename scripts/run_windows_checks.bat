@echo off
setlocal
cd /d "%~dp0\.."
python --version
python -c "import numpy; print('numpy', numpy.__version__)"
python -m unittest discover -s tests -v
if errorlevel 1 exit /b %errorlevel%
python -m ts_pipeline demo --root demo_workspace --seed 20261122 --force
if errorlevel 1 exit /b %errorlevel%
echo Verification reports: demo_workspace\reports
echo Summary: demo_workspace\demo_summary.json
