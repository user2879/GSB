@echo off
cd /d "%~dp0.."
python -m receipt_ledger --ui --sample-dir samples/local --runtime-dir .runtime
