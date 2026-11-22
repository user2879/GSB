$ErrorActionPreference = "Stop"
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    python demo\run_demos.py
} finally {
    Pop-Location
}
