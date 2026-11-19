param(
    [switch]$NoUi
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

python -m spectral_workbench generate --out artifacts\demo\cube
python -m spectral_workbench run `
    --cube artifacts\demo\cube `
    --out artifacts\demo\run `
    --snapshot artifacts\demo\snapshot `
    --version 1 `
    --verify-closure

if (-not $NoUi) {
    python -m spectral_workbench ui --snapshot artifacts\demo\snapshot
}
