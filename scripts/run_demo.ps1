$ErrorActionPreference = "Stop"
$Workspace = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Workspace

py -3.11 -m snapdiag doctor --workspace .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

py -3.11 -m snapdiag demo --workspace .
exit $LASTEXITCODE

