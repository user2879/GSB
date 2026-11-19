$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $Root
try {
    python --version
    python -m narrative_analyzer demo --out-dir artifacts/demo
    python -m narrative_analyzer snapshot-check --snapshot artifacts/demo/snapshot.json
}
finally {
    Pop-Location
}

