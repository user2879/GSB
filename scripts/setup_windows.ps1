$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Get-Command py -ErrorAction SilentlyContinue
if ($null -eq $python) {
    $python = Get-Command python -ErrorAction SilentlyContinue
}
if ($null -eq $python) {
    throw "Python 3.10+ was not found on PATH. Install it from python.org first."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $python.Source -3 -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        & $python.Source -m venv .venv
    }
}

& .venv\Scripts\python.exe -c "import sys; assert sys.version_info >= (3, 10), sys.version"
& .venv\Scripts\python.exe -m pip install --upgrade pip
& .venv\Scripts\python.exe -m pip install -r requirements.txt
& .venv\Scripts\python.exe -m pytest -q
& .venv\Scripts\python.exe -m glyphforge demo --out artifacts\demo --threads 3

Write-Host "Setup and deterministic verification completed." -ForegroundColor Green

