"""可重复演示脚本。

顺序执行：
  1) 一次未压缩训练步（基线，无量化误差）；
  2) 一次压缩训练步（整数定点量化 + 环形 reduce-scatter/all-gather）；
  3) 从压缩快照整体替换恢复后续训，并与不中断基线逐步重放比对。

输出键序稳定的 JSON 总结报告，且同一配置重复运行报告完全一致。
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from gradcomp.config import Config
from gradcomp.data import initial_parameters
from gradcomp.digest import array_digest
from gradcomp.engine import train
from gradcomp.replay import replay_resume
from gradcomp.snapshot import list_snapshots, load_latest

DEMO_WORKDIR = Path("workdir/demo")


def _report_default(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    raise TypeError(type(obj))


def main(transport: str = "loopback") -> dict:
    shutil.rmtree(DEMO_WORKDIR, ignore_errors=True)
    base = Config(workers=4, dimension=64, seed=20261120, samples=64,
                  scale_exponent=2, truncation="floor", block_size=16,
                  base_port=39200, transport=transport, steps=1)

    # 阶段 1：未压缩训练步。
    uncompressed = Config(**{**base.to_dict(), "compressed": False})
    phase1 = train(uncompressed, workdir=DEMO_WORKDIR / "uncompressed")

    # 阶段 2：压缩训练步（连续 2 步，留出快照恢复点）。
    compressed = Config(**{**base.to_dict(), "compressed": True, "steps": 2})
    phase2 = train(compressed, workdir=DEMO_WORKDIR / "compressed")

    # 阶段 3：从最新压缩快照恢复，整体替换后续训 2 步并与基线比对。
    snap, pending = load_latest(DEMO_WORKDIR / "compressed")
    check = replay_resume(compressed, snap, extra_steps=2)

    snap_listing = list_snapshots(DEMO_WORKDIR / "compressed")
    report = {
        "transport": transport,
        "initial_parameter_digest": array_digest(initial_parameters(base)),
        "phase1_uncompressed": {
            "consistent": phase1.steps[0]["consistent"],
            "global_gradient_digest":
                phase1.steps[0]["worker_global_digests"][0],
            "quantization_present": False,
            "parameters_digest": phase1.steps[0]["parameters_digest"],
        },
        "phase2_compressed": {
            "consistent": all(s["consistent"] for s in phase2.steps),
            "global_gradient_digests":
                [s["worker_global_digests"][0] for s in phase2.steps],
            "single_value_bound":
                phase2.steps[0]["error_bounds"]["single_value_bound"],
            "vector_relative_bound": phase2.steps[0]["error_bounds"][
                "vector_relative_bound"]["vector_relative_bound"],
            "dominant_source_share":
                phase2.steps[0]["attribution"]["global_share"],
            "parameters_digest": phase2.final_parameters_digest,
            "parameter_version": phase2.final_parameter_version,
            "snapshot_files": phase2.snapshot_files,
        },
        "phase3_resume_replay": {
            "matched_bit_for_bit": check.matched,
            "start_step": check.start_step,
            "end_step": check.end_step,
            "first_mismatch": check.first_mismatch,
            "pending_or_unknown_steps": pending,
        },
        "snapshot_listing": snap_listing,
        "acceptance": {
            "compressed_global_bytewise_identical":
                all(len(set(s["worker_global_digests"])) == 1
                    for s in phase2.steps),
            "error_bound_verifiable":
                all(s["error_bounds"]["vector_bound_respected"]
                    for s in phase2.steps),
            "resume_replay_bit_identical": check.matched,
        },
    }
    return report


if __name__ == "__main__":
    transport = sys.argv[1] if len(sys.argv) > 1 else "loopback"
    summary = main(transport)
    print(json.dumps(summary, indent=2, sort_keys=True,
                     ensure_ascii=False, default=_report_default))
    ok = all(summary["acceptance"].values())
    sys.exit(0 if ok else 1)
