# Command examples

Set `PY=.venv\Scripts\python.exe` on Windows, or `PY=.venv/bin/python` on Linux/WSL.

```powershell
# Full initial run (or resume when the manifest still matches)
& $PY -m probcal run --config configs/calibration.yaml --state .probcal_state

# Controlled edits; each command increments the snapshot version
& $PY -m probcal edit --config configs/calibration.yaml --state .probcal_state --edit examples/edits/01_switch_and_bins.yaml
& $PY -m probcal edit --config configs/calibration.yaml --state .probcal_state --edit examples/edits/02_regularization.yaml
& $PY -m probcal edit --config configs/calibration.yaml --state .probcal_state --edit examples/edits/03_threshold.yaml
& $PY -m probcal edit --config configs/calibration.yaml --state .probcal_state --edit examples/edits/04_replace_sample.yaml

# Version guard and diagnostics
& $PY -m probcal run --config configs/calibration.yaml --state .probcal_state --expect-version 2
& $PY -m probcal verify --state .probcal_state --output artifacts/verify.json
```

Each edit file accepts a list of operations. Supported operations are
`switch_calibration`, `set_bins`, `set_regularization`, `set_threshold_step`, and
`replace_sample`. Direct edits to `configs/calibration.yaml` after a snapshot has
been committed are rejected with `E_CONFIG_MISMATCH`; use an edit file instead.

