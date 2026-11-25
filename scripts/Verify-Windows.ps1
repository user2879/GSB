$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSScriptRoot)
python -m pip install numpy
python scripts/verify.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host ""
Write-Host "Starting Tkinter workbench..."
python -m scale_review.cli gui
