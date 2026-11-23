$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python -m eventbus_diag validate --spec samples/spec.json --events samples/events.ndjson
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m eventbus_diag run --spec samples/spec.json --events samples/events.ndjson --db artifacts/ledger.sqlite --log artifacts/bus.jsonl --fresh
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m eventbus_diag ui --spec samples/spec.json --events samples/events.ndjson --db artifacts/ledger.sqlite --log artifacts/bus.jsonl
exit $LASTEXITCODE
