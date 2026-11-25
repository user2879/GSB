# 版本标签驱动多节点租约仲裁中间件（首版）

本目录现在包含一套可独立复核的本地实现：声明与事件强校验、真实回环 TCP 子进程节点、显式租约状态机、逐事件/批量双路径仲裁、受双配额约束的线程池调度、SQLite 事务账本、快照整体替换、强杀后恢复重放，以及逐字节对拍工具。

实现仅使用 Python 标准库与本地 `127.0.0.1`，不依赖 Docker、外部网络或云端目标。

## 目录

```text
config/                   节点声明与租约清单
samples/events.jsonl      确定性事件流
lease_arbiter/            实现包
ledgers/                  实测生成的 SQLite 账本
results/                  实测结果与逐字节投影
scripts/run_demo.py       一键验收脚本
tests/                    pytest 自动化测试（26 项）
docs/DESIGN.md            设计、协议、恢复语义
docs/VERIFICATION.md      实际验证记录
```

## 快速开始

```bash
python3 --version                 # 3.11
python3 -m pip install --user numpy pytest

python3 -m lease_arbiter validate \
  --nodes config/nodes.json \
  --leases config/leases.json \
  --events samples/events.jsonl

python3 -m lease_arbiter verify \
  --nodes config/nodes.json \
  --leases config/leases.json \
  --events samples/events.jsonl \
  --out results/verify.json \
  --projection-out results/projection.json

python3 -m lease_arbiter run \
  --nodes config/nodes.json \
  --leases config/leases.json \
  --events samples/events.jsonl \
  --ledger ledgers/manual.sqlite3 \
  --path sequential --fresh \
  --out results/manual-live.json

python3 -m lease_arbiter replay \
  --nodes config/nodes.json \
  --leases config/leases.json \
  --events samples/events.jsonl \
  --ledger ledgers/manual.sqlite3

python3 scripts/run_demo.py       # 两条真实路径 + 独立校验 + 强杀重放
python3 -m pytest
```

`verify` 只接收原始节点声明、租约清单和事件流，不读取账本、运行时缓存或主流程输出。`run` 对已有完整提交的账本要求显式 `--fresh`；恢复核对应使用 `replay`。

## 声明格式

`config/nodes.json`：

```json
{
  "schema_version": 1,
  "declaration_version": 1,
  "global_concurrency": 3,
  "connect_timeout_ms": 1000,
  "read_timeout_ms": 1000,
  "worker_threads": 4,
  "nodes": [
    {"node_id": "alpha", "host": "127.0.0.1", "port": 19341, "max_leases": 2}
  ]
}
```

`config/leases.json`：

```json
{
  "schema_version": 1,
  "declaration_version": 1,
  "leases": [
    {
      "lease_id": "a-cache-1",
      "node_id": "alpha",
      "lease_type": "cache",
      "ttl_ms": 1200,
      "renew_interval_ms": 200,
      "quota_version": 1
    }
  ]
}
```

`events.jsonl` 每行一个事件，`seq` 严格递增且不重复，`at_ms` 单调不减；`grant`/`renew`/`release` 携带 `lease_id` 与 `expected_label`，`quota_invalidate` 携带 `node_id`、`lease_type` 与目标 `quota_version`。

强校验失败统一返回 `E_SPEC`，其中 `field` 为字段路径、`line` 为源文件行号。校验顺序固定，不依赖哈希遍历。

## 关键规则

状态迁移表固定为：

```text
new      -> granted
granted  -> renewed | released | expired
renewed  -> renewed | released | expired
released -> granted
expired  -> granted
```

每次接受迁移必须令标签恰好 `current + 1`；跨代跳跃、重复终态迁移和非法迁移返回 `E_LEASE` 且不改变标签。请求标签与节点当前公认标签不一致返回 `E_STALE`。

默认路径规则是三个条件同时满足时选批量：

```text
event_count >= 10
node_count >= 2
remaining_capacity >= 2
```

结果同时输出阈值、实际值和未触发时距离阈值的边界差值。批量路径按同一时刻帧、`node_id + lease_type` 聚合，帧边界统一处理到期，再按事件序号统一裁决并回填逐事件结局，以保证与逐事件路径一致。

## 错误码

| 错误码 | 含义 |
| --- | --- |
| `E_SPEC` | 声明/事件不合法，附首个非法字段与行号 |
| `E_PORT` | 回环端口被占用或端点不可用 |
| `E_TIMEOUT` | 握手或读取发生真实套接字超时 |
| `E_STALE` | 节点侧标签不匹配，代数不变 |
| `E_LEASE` | 非法迁移、跨代跳跃、终态重复推进 |
| `E_QUOTA` | 全局或节点容量不足，请求不下发节点 |
| `E_DIVERGE` | 双路径首个不一致事件与字段 |
| `E_REPLAY` | 强杀恢复重放与账本首个不一致事件与字段 |
| `E_LEDGER` | 账本事务、快照或会话状态错误 |

## 实际验收结果

- `python3 -m pytest`：`26 passed`。
- 独立校验与两条真实 TCP 路径的投影文件均为 `3538` 字节（规范 JSON `3537` 字节 + 换行），SHA-256 同为 `953b2326d7bee7f4b6ceaa990ac2176c205d54f2271cc5b3fcef973a6f5e598a`。
- 两条真实路径都提交 12 个事件、9 次节点请求，快照版本推进至 13，调度器观测到并行在途峰值 2。
- 强制终止返回 `137`，已完整提交 4 个事件；事件 5 留下 1 个未知待定请求；重放核对成功，恢复快照版本为 5。
- 明细见 `results/demo-report.json`、`results/byte-consistency.json` 与 `docs/VERIFICATION.md`。

## 文档

- `docs/DESIGN.md`：架构、协议、状态机、调度、账本事务与恢复不变量。
- `docs/VERIFICATION.md`：本地执行命令、结果文件和复核方法。
