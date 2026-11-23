#!/usr/bin/env python3
"""独立校验：从零全量重建，逐批次比对快照字节、账本回执与节点期望摘要。

不依赖管线内存状态，仅读取工作目录中的声明、变更集、快照与账本，
验证增量结果与从零全量重建结果逐项一致。退出码非零表示不一致。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfgpipe import sample
from cfgpipe.compiler import compile_all, sha256_hex
from cfgpipe.converge import expected_view
from cfgpipe.graph import node_subscription_map, topo_order
from cfgpipe.pipeline import Pipeline
from cfgpipe.spec import apply_changeset, initial_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()

    pipe = Pipeline(os.path.abspath(args.workdir))
    decl, changesets = pipe.load()
    ledger = pipe.open_ledger()
    session = decl.session
    failures = []

    state = initial_state(decl)
    topo = topo_order(list(state.items),
                      {k: it.derived_from for k, it in state.items.items()})
    info = ledger.recovery_scan()
    committed = {b["batch_id"]: b for b in info.batches}

    for seq in range(0, len(changesets) + 1):
        if seq > 0:
            state = apply_changeset(state, changesets[seq - 1])
        batch_id = "batch_%04d" % seq
        if batch_id not in committed:
            continue
        artifacts = compile_all(state, session.scale_exp, topo)
        snap_dir = os.path.join(pipe.snapshots_dir, batch_id)
        manifest_path = os.path.join(snap_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            failures.append("%s: missing snapshot manifest" % batch_id)
            continue
        with open(manifest_path, "rb") as fh:
            manifest = json.loads(fh.read().decode("utf-8"))
        for key in sorted(manifest):
            item_path = os.path.join(snap_dir, "item_%s.json" % key)
            with open(item_path, "rb") as fh:
                on_disk = fh.read()
            rebuilt = artifacts[key][0]
            if on_disk != rebuilt:
                failures.append("%s/%s: snapshot bytes differ from rebuild"
                                % (batch_id, key))
            if manifest[key] != sha256_hex(rebuilt):
                failures.append("%s/%s: manifest version mismatch"
                                % (batch_id, key))
        # 账本回执 vs 重建期望摘要
        node_subs = node_subscription_map(state)
        receipts = {r["node_id"]: r for r in ledger.receipts_of(batch_id)}
        for node in sorted(decl.nodes, key=lambda n: n.node_id):
            view = expected_view(state.items, node_subs.get(node.node_id, []),
                                 artifacts, session.scale_exp)
            exp = sample.view_digest(view, node.seed, node.sample_size,
                                     session.scale_exp, session.groups)
            got = receipts.get(node.node_id)
            if got is None:
                failures.append("%s/%s: missing ledger receipt"
                                % (batch_id, node.node_id))
            elif got["digest"] != exp:
                failures.append("%s/%s: ledger digest != rebuilt digest"
                                % (batch_id, node.node_id))

    ledger.close()
    if failures:
        print("FULL-REBUILD VERIFICATION FAILED:")
        for line in failures:
            print("  - %s" % line)
        return 1
    print("full-rebuild verification OK: %d committed batches, "
          "all snapshot bytes and ledger digests match a from-scratch rebuild"
          % len(committed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
