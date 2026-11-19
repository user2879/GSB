"""生成确定性样本：初始键值记录与读写轨迹（JSON Lines）。

完全离线、无随机源；重复运行产出逐字节一致的文件。
轨迹覆盖：PUT（含覆盖写）、GET（可见/过期/缺失）、DEL、PURGE、REBUILD_INDEX，
并包含毫秒级过期戳、单调逻辑时钟与连续操作序号。
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def canonical(line: dict) -> str:
    return json.dumps(line, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def initial_records() -> list[dict]:
    return [
        {"key": "user/1/name", "value": "alice", "expire_ms": 0,
         "write_seq": 1, "format_version": 1},
        {"key": "user/1/role", "value": "admin", "expire_ms": 500,
         "write_seq": 1, "format_version": 1},
        {"key": "user/2/name", "value": "bob", "expire_ms": 800,
         "write_seq": 1, "format_version": 1},
        {"key": "cfg/retries", "value": "3", "expire_ms": 0,
         "write_seq": 1, "format_version": 1},
    ]


def trace_operations() -> list[dict]:
    ops: list[dict] = []
    seq = 1

    def add(op: str, clock: int, **fields) -> None:
        nonlocal seq
        entry = {"op_seq": seq, "op": op, "clock": clock}
        entry.update(fields)
        ops.append(entry)
        seq += 1

    # 1-2 新增键（user/3），随后覆盖写同一键（验证覆盖不累积索引）
    add("PUT", 100, key="user/3/name", value="carol", expire_ms=1500)
    add("PUT", 200, key="user/3/name", value="carol2", expire_ms=1500)
    # 3 可见 GET
    add("GET", 250, key="user/3/name")
    # 4 过期 GET（user/1/role expire=500, clock=600）
    add("GET", 600, key="user/1/role")
    # 5 缺失 GET
    add("GET", 650, key="user/9/name")
    # 6 DEL
    add("DEL", 700, key="cfg/retries")
    # 7 PURGE：移除 clock>=expire 的键（user/1/role@500 已过期）
    add("PURGE", 750)
    # 8 重建索引（与前台读写并发的基础场景）
    add("REBUILD_INDEX", 800)
    # 9 重建后再写，验证增量与重建结果共存
    add("PUT", 900, key="user/4/name", value="dave", expire_ms=2000)
    # 10 再新增一个过期键并在到期后 PURGE
    add("PUT", 950, key="temp/session", value="xyz", expire_ms=1000)
    add("GET", 1000, key="temp/session")       # 恰好到期 => 已过期
    add("PURGE", 1100)                         # 物理移除 temp/session
    # 13 第二次重建，验证多次重建稳定
    add("REBUILD_INDEX", 1200)
    # 14-15 收尾查询
    add("GET", 1300, key="user/2/name")        # expire 800 => 已过期
    add("GET", 1300, key="user/1/name")        # 永不过期 => 可见
    return ops


def main() -> None:
    records_path = os.path.join(HERE, "initial_records.json")
    trace_path = os.path.join(HERE, "sample_trace.jsonl")
    with open(records_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(initial_records(), handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    with open(trace_path, "w", encoding="utf-8", newline="\n") as handle:
        for op in trace_operations():
            handle.write(canonical(op) + "\n")
    print(f"wrote {records_path}")
    print(f"wrote {trace_path} ({len(trace_operations())} ops)")


if __name__ == "__main__":
    main()
