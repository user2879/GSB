"""逐字段比对两次运行的最终账本（增量 vs 全量）。

运行时事件（复用/重算/预算审计/快照轨迹/恢复点）属于过程信息，
按业务要求 5 仅核对最终产物：嵌入向量、缓存键、检索顺序、边界样本、
失败清单、待办队列、配置与量化档位。
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vrep.jsonio import canonical, read_json

# 过程性字段：允许两条路径不同，不参与最终一致性判定。
RUNTIME_PATH = ("runtime",)
SNAPSHOT_PATH = ("snapshots",)
RECOVERY_PATH = ("last_recovery",)
GLOBAL_VERSION = ("global_version",)


def comparable_view(ledger: dict) -> dict:
    view = {
        "schema": ledger["schema"],
        "config_hash": ledger["config_hash"],
        "quant_level": ledger["quant_level"],
        "images": ledger["images"],
        "index": ledger["index"],
        "failures": ledger["failures"],
        "pending": ledger["runtime"]["pending"],
    }
    return view


def diff_views(left: dict, right: dict, path: str = "") -> list[str]:
    differences: list[str] = []
    if type(left) is not type(right):
        differences.append(f"{path or '/'}: 类型不同 {type(left).__name__} vs {type(right).__name__}")
        return differences
    if isinstance(left, dict):
        keys = sorted(set(left) | set(right))
        for key in keys:
            if key not in left:
                differences.append(f"{path}/{key}: 仅右侧存在")
            elif key not in right:
                differences.append(f"{path}/{key}: 仅左侧存在")
            else:
                differences.extend(diff_views(left[key], right[key], f"{path}/{key}"))
    elif isinstance(left, list):
        if len(left) != len(right):
            differences.append(
                f"{path or '/'}: 数组长度不同 {len(left)} vs {len(right)}"
            )
        for index, (a, b) in enumerate(zip(left, right)):
            differences.extend(diff_views(a, b, f"{path}[{index}]"))
    else:
        if left != right:
            differences.append(f"{path or '/'}: {left!r} != {right!r}")
    return differences


def main() -> int:
    parser = argparse.ArgumentParser(description="逐字段比对两个账本")
    parser.add_argument("left")
    parser.add_argument("right")
    args = parser.parse_args()

    left_ledger = read_json(args.left)
    right_ledger = read_json(args.right)
    left_view = comparable_view(left_ledger)
    right_view = comparable_view(right_ledger)

    differences = diff_views(left_view, right_view)
    if differences:
        print(f"不一致，共 {len(differences)} 处：")
        for item in differences[:50]:
            print(" -", item)
        return 1

    print("逐字段一致 ✔")
    print(f"  图像数: {len(left_view['images'])}")
    print(f"  索引图像: {len(left_view['index']['images'])}")
    print(f"  边界样本: {left_view['index']['boundary_set']}")
    print(f"  失败清单: {len(left_view['failures'])} 条")
    print(f"  待办队列: {len(left_view['pending'])} 项")
    print(f"  规范化字节数: {len(canonical(left_view))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
