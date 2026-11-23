#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

python -m timbre_workbench demo --out examples/demo
python - <<'PY'
from pathlib import Path
import shutil
path = Path("examples/demo/.workbench")
if path.exists():
    shutil.rmtree(path)
PY

python -m timbre_workbench run \
  --input examples/demo/audio \
  --config examples/demo/config.json \
  --labels examples/demo/labels.json \
  --commands examples/demo/commands.json \
  --work examples/demo/.workbench \
  --report examples/demo/report.json \
  --html examples/demo/report.html \
  --responses examples/demo/responses.json

python -m timbre_workbench compare \
  --input examples/demo/audio \
  --config examples/demo/config.json \
  --labels examples/demo/labels.json \
  --commands examples/demo/commands.json \
  --output examples/demo/compare.json

python -m unittest discover -s tests -v

