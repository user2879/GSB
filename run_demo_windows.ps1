# Windows PowerShell: build venv, install deps, run the headless demo.
# Usage:  .\run_demo_windows.ps1 [-Out artifacts\demo_run]
param(
    [string]$Out = "artifacts\demo_run",
    [int]$Seed = 20261119
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path .venv)) {
    py -3.11 -m venv .venv
}
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Headless execution. Remove this line on a real Windows desktop to show the UI.
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe demo.py --out $Out --seed $Seed
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
