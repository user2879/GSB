$ErrorActionPreference = "Stop"

Write-Host "[1/4] regenerate deterministic samples"
python tools/generate_samples.py

Write-Host "[2/4] preflight"
python -m bucketpack preflight --sqlite-path artifacts/ledger.sqlite

Write-Host "[3/4] tests"
python -m unittest discover -s tests -t . -v

Write-Host "[4/4] acceptance demo"
python -m bucketpack demo --spec samples/spec.json --out-dir artifacts/demo
