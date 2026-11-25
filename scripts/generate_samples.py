"""确定性样本生成入口：python scripts/generate_samples.py [--config ...] [--base-dir ...]"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voicecmd.cli import cmd_generate  # noqa: E402
from voicecmd.config import load_config  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.json")
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()
    cmd_generate(args)
