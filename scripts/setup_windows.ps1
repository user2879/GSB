$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (Test-Path .venv) {
    Remove-Item -Recurse -Force .venv
}

py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) {
    py -3.11 -m venv .venv
}
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.11 or 3.12 is required."
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe scripts\generate_samples.py --out samples
& .\.venv\Scripts\python.exe scripts\run_acceptance.py --workspace .
