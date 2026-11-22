@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
py -3.11 -m qplan.cli recover --ledger workspace\qplan_ledger.sqlite
py -3.11 -m qplan.cli show --ledger workspace\qplan_ledger.sqlite
endlocal
