$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
python examples/check_environment.py
python -m driftwatch validate --snapshots examples/snapshots.json --agents examples/agents_v1.json --criteria examples/criteria.json --check-ports
python examples/run_demo.py --reset
python examples/crash_recovery_demo.py
python -m pytest -q
