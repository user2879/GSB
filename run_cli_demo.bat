@echo off
cd /d "%~dp0"
py -3 run_experiment.py run --spec examples\demo_skewed.json --workdir var\demo --report reports\demo_full_report.json
if errorlevel 1 pause
