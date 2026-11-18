$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
py -3.11 -m mixedprec_diag.cli ui --config configs/float32.json --auto-run @args
exit $LASTEXITCODE
