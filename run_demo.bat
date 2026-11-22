@echo off
REM Windows local deterministic demonstration (no Docker, no network at runtime)
if not exist .venv (
    py -3 -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m gpu_diag.cli init-samples
python -m gpu_diag.cli run
python -m gpu_diag.cli recompute
python -m gpu_diag.cli verify-ledger
python -m gpu_diag.cli ui
