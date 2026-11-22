$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
python scripts/demo.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

