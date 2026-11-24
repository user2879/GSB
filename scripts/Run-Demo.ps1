[CmdletBinding()]
param(
    [string]$Samples = "samples",
    [string]$StateDir = "state/demo",
    [string]$Report = "artifacts/demo_report.json"
)

$ErrorActionPreference = "Stop"

Write-Host "Checking Python dependencies..."
python -c "import sqlite3, tkinter, numpy; print('dependencies ok, numpy', numpy.__version__)"

$Root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = (Join-Path $Root "src") + [IO.Path]::PathSeparator + $env:PYTHONPATH

python (Join-Path $PSScriptRoot "demo.py") `
    --samples $Samples `
    --state-dir $StateDir `
    --report $Report

Write-Host "`nTkinter UI:"
Write-Host "python -m colpage_recovery.cli gui --state-dir $StateDir --report $Report"
