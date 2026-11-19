[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$VenvPython = Join-Path $Root ".venv-win\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    & (Join-Path $Root "setup_offline.ps1")
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$env:PYTHONPATH = $Root
Push-Location $Root
try {
    & $VenvPython -m peft_lora run --samples samples --workspace demo-work-v1
    if ($LASTEXITCODE -ne 0) { throw "v1 demo run failed" }

    & $VenvPython -m peft_lora run --samples samples --workspace demo-work-v1 --base-dir samples/base_v2
    if ($LASTEXITCODE -ne 0) { throw "v2 incremental demo run failed" }

    & $VenvPython -m peft_lora run --samples samples --workspace demo-work-v2-clean --base-dir samples/base_v2
    if ($LASTEXITCODE -ne 0) { throw "v2 clean demo run failed" }

    Write-Host "Demo reports: demo-work-v1\report.json and demo-work-v2-clean\report.json"
} finally {
    Pop-Location
}
