@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
py -3.11 -m qplan.cli spec-check --spec examples\sample_spec.json
if errorlevel 1 exit /b %errorlevel%
py -3.11 -m qplan.cli run --spec examples\sample_spec.json --workspace workspace --batch-size 2 --out workspace\sample_report.json
if errorlevel 1 exit /b %errorlevel%
py -3.11 -m qplan.cli verify --spec examples\sample_spec.json --ledger workspace\qplan_ledger.sqlite --out workspace\sample_verify.json
if errorlevel 1 exit /b %errorlevel%
echo Sample diagnostics and independent verification completed.
endlocal
