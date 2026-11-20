# Demo Input Package

All files are UTF-8 and use a deterministic NumPy seed (`20261121`). Regenerate
with:

```bat
python scripts/generate_samples.py sample_data
```

- `tensors.json`, `samples.jsonl`: tensor manifest and calibration vectors.
- `config.json`: calibration/kernel settings.
- `calibration_v1.txt`, `calibration_v2.json`: exchange packages in both versions.
- `calibration_v2_to_v1_sidecar.json`: reference downgrade-loss metadata.
- `injections.json`: independent injection definitions with stable IDs.

The package deliberately contains symmetric and asymmetric tensors, a heavy-tail
tensor, a constant tensor and an all-zero tensor so clamp, guard and rollback paths
are exercised by the demo.

