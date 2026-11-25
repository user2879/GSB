#!/usr/bin/env python3
"""固定种子生成确定性样本（张量包/配置/v1/v2 权重/命令流/注入配置）。"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fxpdiag.samplegen import SEED, write_samples


def main() -> int:
    parser = argparse.ArgumentParser(description="生成确定性诊断样本")
    parser.add_argument("--samples-dir", default="samples")
    args = parser.parse_args()
    paths = write_samples(args.samples_dir)
    print(f"seed={SEED}")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
