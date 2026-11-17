# Reproducible local demo for Windows PowerShell (no Docker, no network at runtime).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

python -m graymirror.samplegen samples
python -m graymirror.cli run --rules samples/rules.json --stream samples/stream.ndjson `
    --scale-exponent 2 --snapshot samples/snapshots/session-k2.json `
    --report samples/report-k2.json
python -m graymirror.cli restore --rules samples/rules.json `
    --stream samples/stream.ndjson --snapshot samples/snapshots/session-k2.json

foreach ($kind in @("weight_out_of_range","timestamp_regression","snapshot_scale_missing")) {
    # injection is expected to exit 2 with its business code
    python -m graymirror.cli inject --kind $kind
    if ($LASTEXITCODE -ne 2) { throw "expected exit 2 for $kind" }
}
python -m unittest discover -s tests
