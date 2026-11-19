param(
    [string]$Wheelhouse = "wheelhouse",
    [string]$PythonLauncher = "py",
    [string]$PythonVersion = "-3.11"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path $Wheelhouse)) {
    throw "Offline wheel directory not found: $Wheelhouse"
}

& $PythonLauncher $PythonVersion -m venv .venv
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Virtual environment Python not found: $Python"
}

& $Python -m pip install --no-index --find-links $Wheelhouse -r requirements.txt
& $Python -m audiofp doctor --json

Write-Host ""
Write-Host "Offline environment ready. Optional local ffmpeg:"
Write-Host "  `$env:AUDIOFP_FFMPEG = 'C:\tools\ffmpeg\bin\ffmpeg.exe'"
Write-Host "If imageio-ffmpeg is in the wheelhouse, its bundled binary is used automatically."
