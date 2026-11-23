@echo off
setlocal
cd /d "%~dp0"
if not exist data mkdir data
if not exist logs mkdir logs
if not exist snapshots mkdir snapshots
python -m planner_app ui --workspace samples\workspace.json --ledger data\ui.sqlite --snapshots data\snapshots --log logs\ui.jsonl
set CODE=%ERRORLEVEL%
if not "%CODE%"=="0" pause
exit /b %CODE%
