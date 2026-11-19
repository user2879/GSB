[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$VenvPython = Join-Path $Root ".venv-win\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    & (Join-Path $Root "setup_offline.ps1")
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$env:PYTHONPATH = $Root
Push-Location $Root
try {
    & $VenvPython "scripts\run_acceptance.py"
    $ExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $ExitCode
