$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $Root
try {
    python --version
    python scripts/check_env.py
    python -m unittest discover -s tests -v

    $Left = Join-Path $Root "artifacts/verify-a"
    $Right = Join-Path $Root "artifacts/verify-b"
    if (Test-Path $Left) { Remove-Item -Recurse -Force $Left }
    if (Test-Path $Right) { Remove-Item -Recurse -Force $Right }

    python -m narrative_analyzer demo --out-dir $Left
    python -m narrative_analyzer demo --out-dir $Right
    python scripts/compare_dirs.py $Left $Right
    python -m narrative_analyzer snapshot-check --snapshot (Join-Path $Left "snapshot.json")
}
finally {
    Pop-Location
}
