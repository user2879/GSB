#!/usr/bin/env bash
# 端到端演示：生成数据 -> 增量批次 0..3 -> 全量重建 -> 逐字段比对
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m tlc gen-data --out data
for k in 0 1 2 3; do
  python3 -m tlc run --data data --batch "$k" --state state --report "reports/run${k}.json"
done
python3 -m tlc rebuild --data data --batches 3 --state state_full --report reports/rebuild3.json

python3 - << 'PYEOF'
import json
inc = json.load(open('reports/run3.json'))
full = json.load(open('reports/rebuild3.json'))
assert inc == full, '增量运行与全量重建不一致'
print('OK: 增量运行与从零全量重建逐字段一致')
PYEOF
