param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    & $Python "scripts\demo.py"
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
