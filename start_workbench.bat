@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
python -m memdiag.cli gui --plan data\plan.json --samples data\samples.json --db workbench.sqlite3 --snapshots snapshots
endlocal
