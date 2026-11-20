param(
    [string]$Python = "python",
    [string]$Ledger = "state/demo.sqlite",
    [string]$Snapshots = "snapshots"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
New-Item -ItemType Directory -Force artifacts | Out-Null
New-Item -ItemType Directory -Force state | Out-Null
New-Item -ItemType Directory -Force snapshots | Out-Null

$Target1 = Start-Process -FilePath $Python -ArgumentList @("tools/target_stable_gray.py") -PassThru -NoNewWindow
$Target2 = Start-Process -FilePath $Python -ArgumentList @("tools/target_mirror.py") -PassThru -NoNewWindow
try {
    Start-Sleep -Milliseconds 700
    & $Python -m gray_arbiter run `
        --config samples/config.json `
        --requests samples/requests.ndjson `
        --ledger $Ledger `
        --snapshots $Snapshots `
        --reset-ledger `
        --batch-size 36 `
        --workers 16 | Tee-Object -FilePath "artifacts/demo-report.json"
    & $Python -m gray_arbiter replay --ledger $Ledger
    & $Python -m gray_arbiter ui --ledger $Ledger --snapshots $Snapshots
}
finally {
    Stop-Process -Id $Target1.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $Target2.Id -Force -ErrorAction SilentlyContinue
}
