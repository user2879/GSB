$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

Write-Host "[1/7] Install dependencies"
py -3.11 -m pip install -r requirements.txt

Write-Host "[2/7] Generate deterministic local samples"
py -3.11 tools/generate_samples.py

Write-Host "[3/7] CLI, log directory, NumPy, SQLite, Tk preflight"
py -3.11 -m vecflow preflight --log-dir logs

Write-Host "[4/7] Strong validation and deterministic plan"
py -3.11 -m vecflow validate --log-dir logs

Write-Host "[5/7] Normal budgeted parallel execution (ledger and snapshot reset)"
py -3.11 -m vecflow run --reset-state --log-dir logs

Write-Host "[6/7] Forced-termination recovery and replay demo"
py -3.11 -m vecflow run --ledger artifacts/crash_ledger.sqlite --snapshot-dir artifacts/crash_snapshots --log-dir logs --reset-state --force-crash-after 2
py -3.11 -m vecflow recover --ledger artifacts/crash_ledger.sqlite --log-dir logs
py -3.11 -m vecflow replay --ledger artifacts/crash_ledger.sqlite --snapshot-dir artifacts/crash_snapshots --log-dir logs

Write-Host "[7/7] Launch Tkinter workbench"
py -3.11 -m vecflow gui --log-dir logs
