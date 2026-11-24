# Windows PowerShell repeatable demo (run from the repository root)
.\.venv\Scripts\Activate.ps1
python -m batchdiag.cli gen-samples --samples-dir samples
python -m batchdiag.cli validate --requests samples/requests.json --workers samples/workers.json --rules samples/merge_rules.json
python -m batchdiag.cli normalize --requests samples/requests.json --workers samples/workers.json --rules samples/merge_rules.json --receipts samples/receipts_r3.json
python -m batchdiag.cli run --requests samples/requests.json --workers samples/workers.json --rules samples/merge_rules.json --ledger artifacts/ledger.sqlite3
python -m batchdiag.cli scheduler --requests samples/requests.json --workers samples/workers.json --rules samples/merge_rules.json
python -m batchdiag.cli verify --requests samples/requests.json --workers samples/workers.json --rules samples/merge_rules.json --ledger-a artifacts/a.sqlite3 --ledger-b artifacts/b.sqlite3
