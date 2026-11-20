"""生成本地确定性样本（v1/v2/v3 清单、节点注册表、样本数据）。

用法: python scripts/gen_samples.py [workdir]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configdist.cli import main

if __name__ == "__main__":
    workdir = sys.argv[1] if len(sys.argv) > 1 else "demo_ws"
    sys.exit(main(["gen-samples", "--workdir", workdir]))
