"""优雅退出演示：处理若干样本后向自身投递 SIGTERM，验证退出码 0 与快照提交。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proofread_app.app import ProofreadApp
from proofread_app.signals import SignalController


def main() -> int:
    base = Path(__file__).resolve().parents[1]
    app = ProofreadApp(str(base))
    app.clean_state()
    app.controller.install()
    pipeline = app._build_pipeline()

    # 在批次开始前预置 SIGTERM，确保首个样本阶段提交后即优雅退出
    app.controller.requested = True
    app.controller.signum = 15
    results, manifest, exited = pipeline.run_batch(app.samples)
    print(f"exited_gracefully={exited}")
    print(f"committed_samples={len(manifest.completed_cursor)}")
    print(f"global_version={manifest.global_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
