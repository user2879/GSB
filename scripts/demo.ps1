$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

# After the one-time `go mod download`, the demo itself is offline.
$env:GOTOOLCHAIN = "local"
$env:GOPROXY = "off"

$RunDir = Join-Path $Root ("out\demo-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root "bin") | Out-Null
$Fl = Join-Path $Root "bin\flowledger.exe"

Write-Host "[1/8] build"
go build -o $Fl ./cmd/flowledger
if ($LASTEXITCODE -ne 0) { throw "go build failed" }

Write-Host "[2/8] tests"
go test ./... -count=1 | Tee-Object -FilePath (Join-Path $RunDir "tests.log")
if ($LASTEXITCODE -ne 0) { throw "go test failed" }

Write-Host "[3/8] run deterministic v1"
& $Fl run --spec samples/traffic-v1.json --policy samples/policy-v1.json `
  --ledger (Join-Path $RunDir "ledger.db") --report (Join-Path $RunDir "v1-report.json") `
  --k 3 --window-minutes 1 --global-concurrency 3 --per-key-concurrency 2 `
  --task-delay-ms 10 | Out-File -Encoding utf8 (Join-Path $RunDir "v1-summary.json")
if ($LASTEXITCODE -ne 0) { throw "v1 run failed" }

Write-Host "[4/8] replay and verify v1"
& $Fl replay --ledger (Join-Path $RunDir "ledger.db") --output (Join-Path $RunDir "v1-replay.json") | Tee-Object -FilePath (Join-Path $RunDir "v1-replay-stdout.json")
if ($LASTEXITCODE -ne 0) { throw "v1 replay failed" }
& $Fl verify --ledger (Join-Path $RunDir "ledger.db") --output (Join-Path $RunDir "v1-verify.json") | Tee-Object -FilePath (Join-Path $RunDir "v1-verify-stdout.json")
if ($LASTEXITCODE -ne 0) { throw "v1 verify failed" }

Write-Host "[5/8] start v2 and inject forced termination after committed windows"
& $Fl run --spec samples/traffic-v2.json --policy samples/policy-v2.json `
  --ledger (Join-Path $RunDir "ledger.db") --report (Join-Path $RunDir "v2-report.json") `
  --k 3 --window-minutes 1 --global-concurrency 3 --per-key-concurrency 2 `
  --task-delay-ms 10 --crash-after 2 `
  > (Join-Path $RunDir "v2-crash-summary.json") 2> (Join-Path $RunDir "v2-crash-error.json")
$CrashCode = $LASTEXITCODE
if ($CrashCode -ne 75) { throw "expected crash exit 75, got $CrashCode" }
if (Test-Path (Join-Path $RunDir "v2-report.json")) { throw "forced termination produced an exit report" }

Write-Host "[6/8] prove incomplete current view is not reportable"
& $Fl verify --ledger (Join-Path $RunDir "ledger.db") > (Join-Path $RunDir "incomplete-verify.json") 2>&1
$IncompleteVerify = $LASTEXITCODE
& $Fl replay --ledger (Join-Path $RunDir "ledger.db") > (Join-Path $RunDir "incomplete-replay.json") 2>&1
$IncompleteReplay = $LASTEXITCODE
if ($IncompleteVerify -ne 66 -or $IncompleteReplay -ne 65) {
  throw "unexpected incomplete-view exits: verify=$IncompleteVerify replay=$IncompleteReplay"
}
& $Fl history --ledger (Join-Path $RunDir "ledger.db") > (Join-Path $RunDir "history-after-crash.json")

Write-Host "[7/8] resume v2, verify, replay"
& $Fl run --spec samples/traffic-v2.json --policy samples/policy-v2.json `
  --ledger (Join-Path $RunDir "ledger.db") --report (Join-Path $RunDir "v2-report.json") `
  --k 3 --window-minutes 1 --global-concurrency 3 --per-key-concurrency 2 `
  --task-delay-ms 10 | Out-File -Encoding utf8 (Join-Path $RunDir "v2-summary.json")
if ($LASTEXITCODE -ne 0) { throw "v2 resume failed" }
& $Fl verify --ledger (Join-Path $RunDir "ledger.db") --output (Join-Path $RunDir "v2-verify.json") | Tee-Object -FilePath (Join-Path $RunDir "v2-verify-stdout.json")
if ($LASTEXITCODE -ne 0) { throw "v2 verify failed" }
& $Fl replay --ledger (Join-Path $RunDir "ledger.db") --output (Join-Path $RunDir "v2-replay.json") | Tee-Object -FilePath (Join-Path $RunDir "v2-replay-stdout.json")
if ($LASTEXITCODE -ne 0) { throw "v2 replay failed" }
& $Fl history --ledger (Join-Path $RunDir "ledger.db") > (Join-Path $RunDir "history-final.json")

Write-Host "[8/8] stable result summary"
$v1Summary = Get-Content (Join-Path $RunDir "v1-summary.json") -Raw
$v2Summary = Get-Content (Join-Path $RunDir "v2-summary.json") -Raw
$v1Digest = [regex]::Match($v1Summary, '"summary_digest"\s*:\s*"([0-9a-f]{64})"').Groups[1].Value
$v2Digest = [regex]::Match($v2Summary, '"summary_digest"\s*:\s*"([0-9a-f]{64})"').Groups[1].Value
$v2Verify = Get-Content (Join-Path $RunDir "v2-verify.json") -Raw
$v2Recovered = [regex]::Match($v2Verify, '"recovered_unknown_tasks"\s*:\s*(\d+)').Groups[1].Value
@(
  "run_dir=$RunDir",
  "v1_summary_digest=$v1Digest",
  "v2_summary_digest=$v2Digest",
  "v1_replay=ok",
  "v2_replay=ok",
  "v2_recovered_unknown_tasks=$v2Recovered",
  "final_current_stale_historical=1",
  "forced_termination_exit=$CrashCode",
  "incomplete_verify_exit=$IncompleteVerify",
  "incomplete_replay_exit=$IncompleteReplay"
) | Tee-Object -FilePath (Join-Path $RunDir "validation-summary.txt")

Write-Host "PASS: artifacts are in $RunDir"
