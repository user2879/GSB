param(
    [int]$Threads = 3,
    [string]$Out = "artifacts\demo"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Local virtual environment missing. Run scripts\setup_windows.ps1 first."
}

& .venv\Scripts\python.exe -m glyphforge demo --out $Out --threads $Threads
if ($LASTEXITCODE -ne 0) {
    throw "GlyphForge demo failed with exit code $LASTEXITCODE"
}

