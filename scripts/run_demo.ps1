param(
    [string]$Seed = "20261119",
    [string]$Output = "artifacts\demo"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Run scripts\setup_windows_offline.ps1 first."
}

& $Python -m audiofp demo --out $Output --seed $Seed
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Report: $Output\reports\demo_report.md"
Write-Host "JSON:   $Output\reports\demo_report.json"
Write-Host "Repeat: $Output\reports\repeatability.json"
