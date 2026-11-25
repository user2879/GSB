#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON:-.venv/bin/python}"
mkdir -p artifacts/verification
"${PYTHON_BIN}" -m unittest discover -v 2>&1 | tee artifacts/verification/tests.log
"${PYTHON_BIN}" -m tile_anomaly run --config sample/config/detection.yaml --reset --json \
  > artifacts/verification/baseline_run.json
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import shutil
from tile_anomaly.edits import clone_sample
source = Path('sample')
destination = Path('artifacts/verification/fault-sample')
if destination.exists():
    shutil.rmtree(destination)
clone_sample(source, destination)
PY
"${PYTHON_BIN}" -m tile_anomaly run \
  --config artifacts/verification/fault-sample/config/detection.yaml \
  --fault-config artifacts/verification/fault-sample/config/fault_injection.json \
  --reset --json \
  > artifacts/verification/fault_run.json
"${PYTHON_BIN}" - <<'PY'
import json
import platform
import sys
from pathlib import Path

import numpy
import PIL

root = Path('artifacts/verification')


def summarize(name):
    result = json.loads((root / f'{name}_run.json').read_text(encoding='utf-8'))
    snapshot = Path(result['snapshot_dir'])
    pixel = json.loads((snapshot / 'pixel_map.json').read_text(encoding='utf-8'))
    diagnostics = json.loads((snapshot / 'diagnostics.json').read_text(encoding='utf-8'))
    current_diagnostics = diagnostics['by_key'][diagnostics['current_key']]
    return {
        'complete': result['complete'],
        'global_version': result['global_version'],
        'stage_content_keys': result['run_manifest']['stage_content_keys'],
        'scheduler_budget': result['run_manifest']['scheduler_budget'],
        'audit_summary': result['audit_summary'],
        'events': result['events'],
        'pixel': {
            'anomaly': pixel['anomaly_pixel_count'],
            'normal': pixel['normal_pixel_count'],
            'unknown': pixel['unknown_pixel_count'],
            'fusion_band_count': len(pixel['fusion_band_ids']),
            'postprocess': pixel['postprocess'],
        },
        'diagnostics': {
            'boundary_count': current_diagnostics['boundary_count'],
            'flip_count': current_diagnostics['flip_count'],
            'higher_width_fraction_bits': current_diagnostics['higher_width_fraction_bits'],
        },
    }

test_log = (root / 'tests.log').read_text(encoding='utf-8')
summary = {
    'environment': {
        'linux': platform.platform(),
        'python': sys.version.split()[0],
        'numpy': numpy.__version__,
        'pillow': PIL.__version__,
    },
    'tests_ok': 'OK' in test_log,
    'baseline': summarize('baseline'),
    'fault': summarize('fault'),
}
(root / 'summary.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
    encoding='utf-8',
)
PY
