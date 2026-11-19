$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python run_demo.py --workspace (Join-Path $PSScriptRoot "demo_workspace") @args
exit $LASTEXITCODE
