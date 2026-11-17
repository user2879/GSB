param(
    [string]$Python = "py -3.11",
    [string]$Config = "data/demo_routes.json",
    [string]$StateDir = "artifacts/demo-state"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Invoke-Expression "$Python scripts/cleanup_ports.py --config `"$Config`" --state-dir `"$StateDir`""
Invoke-Expression "$Python scripts/run_demo.py --config `"$Config`" --state-dir `"$StateDir`""
