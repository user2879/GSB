#!/usr/bin/env bash
# 本地确定性演示：强校验 -> 两次构建(逐字节一致) -> 失败/超时/淘汰 -> 快照重放
set -euo pipefail
cd "$(dirname "$0")"
SRC=samples/workspace
MAN=samples/manifests/sample_manifest.json
WD=workdir/demo
python3 -m layerbench.cli validate --manifest "$MAN" --workdir "$WD" --source-root "$SRC"
python3 -m layerbench.cli verify   --manifest "$MAN" --workdir "$WD" --source-root "$SRC"
SNAP=$(ls -d "$WD"/snapshots/build-* | grep -v resumed | tail -1)
python3 -m layerbench.cli replay --snapshot "$SNAP" --workdir "$WD" --source-root "$SRC"
