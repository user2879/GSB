$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
}

& ".venv\Scripts\python.exe" -m pip install --upgrade pip
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt
& ".venv\Scripts\python.exe" -c "import numpy, PySide6; print('numpy', numpy.__version__); print('PySide6', PySide6.__version__)"
& ".venv\Scripts\python.exe" -m mocap_workbench demo --workspace . --frames 1200

Write-Host ""
Write-Host "Setup complete. Launch with:"
Write-Host "  .\\.venv\\Scripts\\python.exe -m mocap_workbench app --workspace . --frames 1200"

