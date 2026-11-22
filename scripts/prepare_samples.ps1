$ErrorActionPreference = "Stop"
$Workspace = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Workspace
py -3.11 -m snapdiag prepare-samples --workspace .
exit $LASTEXITCODE

