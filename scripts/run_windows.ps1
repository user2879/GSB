[CmdletBinding()]
param(
    [switch]$SkipInstall,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.11 -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version; print(sys.version)"
    $VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
    if (-not (Test-Path $VenvPython)) {
        & py -3.11 -m venv (Join-Path $Root ".venv")
    }
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version; print(sys.version)"
    $VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
    if (-not (Test-Path $VenvPython)) {
        & python -m venv (Join-Path $Root ".venv")
    }
} else {
    throw "Python 3.11 was not found. Install Python 3.11 and retry."
}

if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment Python was not created: $VenvPython"
}

if (-not $SkipInstall) {
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -e ".[test]"
}

$env:PYTHONPATH = (Join-Path $Root "src") + $(if ($env:PYTHONPATH) { ";" + $env:PYTHONPATH } else { "" })

& $VenvPython -m haptic_diag.cli generate
if (-not $SkipTests) {
    & $VenvPython -m pytest -q
}

$Workbench = Join-Path $Root "workbench"
$Report = Join-Path $Root "reports\verification-windows.json"
& $VenvPython -m haptic_diag.cli verify `
    --sample cross_slice_boundary `
    --protocol 2 `
    --workspace $Workbench `
    --output $Report `
    --batch-size 20

Write-Host "Verification report: $Report"
