@echo off
setlocal
cd /d %~dp0
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
python -m qos_diag sample --workspace .
if errorlevel 1 exit /b 1
python -m qos_diag drill --workspace . --output artifacts\reports\acceptance-drill.json
if errorlevel 1 exit /b 1
python -m unittest discover -s tests
if errorlevel 1 exit /b 1
python -m qos_diag verify --spec-dir samples\edited_param --ledger artifacts\ledger\qos.db
exit /b %errorlevel%
