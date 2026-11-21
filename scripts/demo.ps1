$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
python scripts/check_env.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/run_demo.py @args
exit $LASTEXITCODE
