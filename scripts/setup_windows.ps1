$ErrorActionPreference = "Stop"

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .

Write-Host "Environment ready. Activate with: .venv\Scripts\Activate.ps1"
Write-Host "Run tests with: .venv\Scripts\python.exe -m pytest -q"
Write-Host "Open UI with: .venv\Scripts\python.exe -m tcp_workbench.cli ui --reset --config samples/session.json"

