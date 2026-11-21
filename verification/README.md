# Verification Evidence

- `test_results.txt` — output of `python -m unittest discover -s tests -v`.
- `clean_run_summary.json` — condensed result of `reports/full_report.json`.
- `06_full_rebuild.json` — full rebuild of the final edited demo dataset.
- `07_parity_incremental_vs_full.json` — edited incremental run vs full rebuild.
- `09_parity_repeat_vs_full.json` — unchanged incremental rerun vs full rebuild.
- `12_parity_resumed_vs_full.json` — snapshot interrupt/resume vs full rebuild.
- `14_parity_concurrency_1_vs_4.json` — concurrency 1 vs 4.
- `15_clean_repeat_compare.json` — clean full rerun vs saved clean report.
- `demo_summary.json` — all demo scenarios, reuse counts, injection mapping, quota/budget codes.

All listed parity reports contain `"equal": true` and `"difference_count": 0`. The demo can be regenerated with:

```bat
python -m ts_pipeline demo --root demo_workspace --seed 20261122 --force
```
