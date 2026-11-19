$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Local virtual environment missing. Run scripts\setup_windows.ps1 first."
}

& .venv\Scripts\python.exe -m glyphforge gui

