param([switch]$Offline)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

py -3.11 -m venv .venv
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if ($Offline) {
    & $Python -m pip install --no-index --find-links (Join-Path $Root "wheelhouse") -r requirements.txt
} else {
    & $Python -m pip install --disable-pip-version-check -r requirements.txt
}

Write-Host "Environment ready: $Python"
