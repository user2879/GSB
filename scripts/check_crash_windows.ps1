[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Run scripts\run_windows.ps1 first so the virtual environment exists."
}
$env:PYTHONPATH = (Join-Path $Root "src") + $(if ($env:PYTHONPATH) { ";" + $env:PYTHONPATH } else { "" })
$Workspace = Join-Path $Root "workbench-crash"

& $Python -m haptic_diag.cli crash-worker --workspace $Workspace --crash-after-batches 2 --commit-crash-after-files 3
if ($LASTEXITCODE -eq 0) {
    throw "crash-worker unexpectedly exited successfully"
}

& $Python -m haptic_diag.cli recover --workspace $Workspace
if ($LASTEXITCODE -ne 0) {
    throw "recover failed after forced termination"
}

& $Python -m pytest -q tests\test_persistence.py::test_real_subprocess_forced_termination_then_resume_matches_full
if ($LASTEXITCODE -ne 0) {
    throw "post-crash resume/full-compile comparison failed"
}

