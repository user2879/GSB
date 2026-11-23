@echo off
setlocal
cd /d "%~dp0"
python -c "import tkinter, numpy, sqlite3" || goto :error
if not exist "data\demo_large.json" python -m mvcc_workbench init
python -m mvcc_workbench gui --spec "data\demo_large.json" --ledger "data\ledger.sqlite3"
exit /b %errorlevel%

:error
echo Environment check failed: tkinter, numpy or sqlite3 is unavailable.
exit /b 1
