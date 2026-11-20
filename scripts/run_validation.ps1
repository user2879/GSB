$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $root
try {

& $python -m pointcloud_studio generate --out (Join-Path $root "artifacts\sample")
& $python -m pointcloud_studio validate `
    --sample (Join-Path $root "artifacts\sample") `
    --workspace (Join-Path $root "artifacts\run_workspace") `
    --report (Join-Path $root "artifacts\validation_report.json")

if ($LASTEXITCODE -ne 0) {
    throw "Validation failed with exit code $LASTEXITCODE"
}
}
finally {
    Pop-Location
}
