$ErrorActionPreference = "Stop"
python --version
python -m pip install -r requirements-dev.txt
python -m receipt_ledger --generate-samples --sample-dir samples/local
python -m receipt_ledger --verify --sample-dir samples/local --runtime-dir .runtime
python -m pytest -q tests/test_receipt_ledger.py
python -m receipt_ledger --inject spec --runtime-dir .runtime
python -m receipt_ledger --inject ref --runtime-dir .runtime
python -m receipt_ledger --inject snapshot --runtime-dir .runtime
Write-Host "Verification complete. See artifacts/verification-result.json and .runtime/logs/injections.jsonl"
