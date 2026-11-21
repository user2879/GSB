# Windows PowerShell: full deterministic verification sequence.
.\.venv\Scripts\Activate.ps1
python -m hotgray.samples --out-dir samples
python -m hotgray validate samples/gateway_declaration.json
python scripts/demo_e2e.py
python scripts/demo_recovery.py
python scripts/demo_faults.py
python -m pytest -q
