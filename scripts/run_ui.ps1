param(
    [string]$Index = "artifacts\demo\index\current.db",
    [switch]$Portable,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Run scripts\setup_windows_offline.ps1 first."
}

$Arguments = @("-m", "audiofp", "ui", "--index", $Index)
if ($Portable) { $Arguments += "--portable" }
if ($Smoke) { $Arguments += "--smoke" }
& $Python @Arguments
