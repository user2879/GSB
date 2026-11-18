$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
py -3.11 -m mixedprec_diag.cli demo --config-dir configs --report-dir reports --snapshot-dir snapshots @args
exit $LASTEXITCODE
