param(
    [string]$Workspace = "workspace-demo"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Run scripts\setup_windows.ps1 first."
}

.\.venv\Scripts\python.exe -m forensic_workbench demo --workspace $Workspace

