$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3 -m venv .venv
    } else {
        python -m venv .venv
    }
}

& ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
& ".venv\Scripts\python.exe" -m spatial_diag run --workspace artifacts

Write-Host "`nDiagnostics committed under artifacts\."
Write-Host "Verify with: .venv\Scripts\python.exe -m spatial_diag verify --workspace artifacts"
