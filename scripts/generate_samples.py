"""生成确定性样本清单到 samples/generated。"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audit_engine.generator import generate_all

if __name__ == "__main__":
    written = generate_all("samples/generated")
    print(f"已生成 {len(written)} 份确定性清单：")
    for path in written:
        print(" -", path)
