#!/usr/bin/env python3
"""生成固定种子合成样本数据。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tlc.datagen import generate

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data"
    generate(out)
    print(f"样本数据已生成: {out}")
