@echo off
setlocal
cd /d "%~dp0"
if not exist data mkdir data
if not exist logs mkdir logs
if not exist snapshots mkdir snapshots
python -m planner_app demo --reset-ledger --workspace samples\workspace.json --ledger data\demo.sqlite --snapshots data\snapshots --log logs\demo.jsonl --report artifacts\verification-report.json
set CODE=%ERRORLEVEL%
echo.
echo Demo exit code: %CODE%
pause
exit /b %CODE%
