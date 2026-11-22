$ErrorActionPreference = "Stop"
python app.py demo --workspace demo_workspace --reset
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m pytest -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python verify_acceptance.py
