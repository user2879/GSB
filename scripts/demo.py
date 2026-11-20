"""端到端验收演示：归一、优先级下发、收敛、权威重算、强杀恢复。

用法: python scripts/demo.py [workdir]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configdist.cli import main

if __name__ == "__main__":
    workdir = sys.argv[1] if len(sys.argv) > 1 else "demo_ws"
    sys.exit(main(["demo", "--workdir", workdir]))
