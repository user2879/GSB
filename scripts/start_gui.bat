@echo off
setlocal
cd /d "%~dp0\.."
if not exist .venv\Scripts\python.exe (
  echo Create .venv and install the project first. See README.md.
  exit /b 1
)
.venv\Scripts\python.exe -m gateway_diag gui --spec examples\sample_spec.json --requests examples\sample_requests.ndjson --db artifacts\ledger.sqlite3
