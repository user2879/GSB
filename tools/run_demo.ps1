$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

Write-Host "== Python =="
python --version
python -c "import numpy; print('numpy', numpy.__version__)"

Write-Host "== Generate deterministic samples =="
python tools/build_samples.py

Write-Host "== Replay command stream and commit snapshots =="
python -m opexlab replay --reset

Write-Host "== Convert v1 -> v2 =="
python -m opexlab convert --to v2 --output runtime/conversions/weights_v2_from_v1.jsonl

Write-Host "== Convert v2 -> v1 =="
python -m opexlab convert --to v1 --output runtime/conversions/weights_v1_from_v2.txt

Write-Host "== Isolated fault injections =="
python -m opexlab inject

Write-Host "== Acceptance suite =="
python -m opexlab verify

Write-Host "== Unit tests =="
python -m unittest discover -s tests -v
