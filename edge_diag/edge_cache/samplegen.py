"""生成演示用声明、内容清单与事件流样本。

样本设计（确定性、可复跑）：
- 3 节点，全局并发上限 4，制造真实排队，且每个排队者都能被后续释放解锁
- 2 个内容，多个分片
- 事件覆盖：授予排队、读取、续约成功、版本冲突续约(E_LEASE)、
  真实单调时钟到期回收、释放、取消、配额换代回收(与到期交错)、
  预算耗尽(E_BUDGET)
- 到期依赖真实 NumPy 校验 + TCP 往返的自然耗时，不使用任何人为 sleep
"""

import json
from pathlib import Path


def build_sample():
    spec = {
        "session": {
            "session_id": "edge-demo-001",
            "global_lease_limit": 4,
            "global_byte_budget": 2048,
            "session_scale": 1000,
        },
        "nodes": [
            {"node_id": "node-a", "host": "127.0.0.1", "port": 19001,
             "capacity_slots": 3, "per_node_lease_limit": 2,
             "node_seed": 101, "quota_version": 7},
            {"node_id": "node-b", "host": "127.0.0.1", "port": 19002,
             "capacity_slots": 2, "per_node_lease_limit": 2,
             "node_seed": 202, "quota_version": 7},
            {"node_id": "node-c", "host": "127.0.0.1", "port": 19003,
             "capacity_slots": 2, "per_node_lease_limit": 2,
             "node_seed": 303, "quota_version": 3},
        ],
        "contents": [
            {"content_key": "video-1080", "total_bytes": 1024,
             "shard_count": 4, "content_seed": 9001},
            {"content_key": "index-meta", "total_bytes": 512,
             "shard_count": 2, "content_seed": 9002},
        ],
        "events": [
            # 先占满全局 4 个名额（node-a×2, node-b, node-c），长 TTL
            {"seq": 1, "type": "write", "node_id": "node-a",
             "content_key": "video-1080", "shard_index": 0, "ttl_ms": 30000},
            {"seq": 2, "type": "write", "node_id": "node-a",
             "content_key": "video-1080", "shard_index": 1, "ttl_ms": 30000},
            {"seq": 3, "type": "write", "node_id": "node-b",
             "content_key": "video-1080", "shard_index": 2, "ttl_ms": 30000},
            {"seq": 4, "type": "write", "node_id": "node-c",
             "content_key": "index-meta", "shard_index": 0, "ttl_ms": 30000},
            # seq5 到达时全局已满 -> 真实排队；seq6 释放 node-a 名额解锁
            {"seq": 5, "type": "write", "node_id": "node-a",
             "content_key": "index-meta", "shard_index": 0, "ttl_ms": 30000},
            {"seq": 6, "type": "release", "node_id": "node-a",
             "content_key": "video-1080", "shard_index": 1},
            {"seq": 7, "type": "read", "node_id": "node-a",
             "content_key": "video-1080", "shard_index": 0},
            # 续约成功（乐观版本 1->2）
            {"seq": 8, "type": "renew", "node_id": "node-b",
             "content_key": "video-1080", "shard_index": 2, "ttl_ms": 30000},
            # 配额换代：node-c 旧配额 3 的活跃租约被配额回收
            {"seq": 9, "type": "quota_change", "node_id": "node-c",
             "new_quota_version": 4},
            # 配额回收后以新版本 4 重新授予一个短 TTL 租约（30ms）
            {"seq": 10, "type": "write", "node_id": "node-c",
             "content_key": "video-1080", "shard_index": 3, "ttl_ms": 30},
            # 紧随的真实读取批次（NumPy 定点校验 + TCP 往返，自然耗时）
            # 使 seq10 租约在 seq12 处理前由真实单调时钟到期
            {"seq": 11, "type": "read", "node_id": "node-a",
             "content_key": "video-1080", "shard_index": 0},
            {"seq": 12, "type": "read", "node_id": "node-c",
             "content_key": "video-1080", "shard_index": 3},
            # 取消一个长 TTL 租约
            {"seq": 13, "type": "cancel", "node_id": "node-a",
             "content_key": "index-meta", "shard_index": 0},
            # 先释放再对失效租约续约 -> E_LEASE（乐观校验拒绝且不延期）
            {"seq": 14, "type": "release", "node_id": "node-b",
             "content_key": "video-1080", "shard_index": 2},
            {"seq": 15, "type": "renew", "node_id": "node-b",
             "content_key": "video-1080", "shard_index": 2, "ttl_ms": 500},
            # 预算耗尽：累计下发字节越过 2048 -> E_BUDGET
            {"seq": 16, "type": "write", "node_id": "node-b",
             "content_key": "index-meta", "shard_index": 1, "ttl_ms": 500},
            # 到期/回收后的只读复核
            {"seq": 17, "type": "read", "node_id": "node-a",
             "content_key": "video-1080", "shard_index": 0},
            # 配额再次换代，与到期裁决交错
            {"seq": 18, "type": "quota_change", "node_id": "node-b",
             "new_quota_version": 8},
        ],
    }
    return spec


def write_sample(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_sample(), ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path
