@echo off
REM Open the Tkinter diagnostics window (falls back to text when tk is absent).
python -m slowprof.cli diagnose --samples data\samples.ndjson --ledger state\ledger.sqlite --snapshot-dir state\snap %*
