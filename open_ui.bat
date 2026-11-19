@echo off
setlocal
cd /d "%~dp0"
python -m srebudget.cli generate --data-dir data
python -m srebudget.cli run --data-dir data --session-id ui-session --batch-size 4 --clock trace --snapshot
python -m srebudget.cli ui --data-dir data --session-id ui-session
endlocal
