# 实际验证记录

验证日期：2026-11-21（Asia/Shanghai）。所有操作在本机工作区执行，未使用 Docker，未访问外部网络目标。

## 环境

```text
Python       3.11.0rc1
SQLite       3.37.2
NumPy        1.26.4
socket       available
tkinter      unavailable: ModuleNotFoundError（当前容器；核心链路不需要）
loopback     127.0.0.1 高位端口探测成功（49152）
```

复现：

```bash
python3 -m replica_repair env
```

## 自动化测试

```bash
python3 -m unittest discover -s tests -v
```

最终结果：29 项测试全部通过，覆盖：

- v1/v2/v3 严格编码/解码、字段路径和非法摘要/大小；
- v1 升级强制终止后按检查点前缀续做，已提交节点不重复读取；
- v2 升级、只读备份、`manifest_version` 单调递增；
- 节点摘要查询、数据拉取、进程强杀重启、截断 `E_BLOCK`、内容损坏 `E_INTEGRITY`；
- 频次/优先级/节点标识稳定真值规则及重复输出逐字节一致；
- 配额排队、每轮并发/节点 KB 上限、原子修复写入；
- 前后快照、成本、账本格式版本/内容摘要换代、未完成任务 `unknown`；
- 修复进程强制终止后的清单对账和再次完成；
- 派生关系/清单版本演进后的陈旧影响和整体视图替换；
- 账本字段篡改、块字节篡改的 `E_REPLAY` 首差异；
- v1/v2/v3 全链路 demo、CLI run/report/replay、两次 demo 确定性比较。

## 全链路 demo 摘要

```bash
python3 -m replica_repair demo --root .
```

退出码 `0`，`data/demo_report.json` 中的关键结果：

| 组 | 源版本 | 最终版本 | 修复前失配 | 修复任务 | 修复 KB | 成本 | 队列事件 | 重放 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| interop-v1 | 1 | 3 | 2 | 2 | 5 | 5 | 0 | 通过 |
| interop-v2 | 2 | 3 | 1 | 1 | 2 | 2 | 0 | 通过 |
| interop-v3 | 3 | 2 | 4 | 4 | 10 | 10 | 1（`per_node_quota`） | 通过 |

说明：表中 v3 的“最终版本 2”指首次修复后的清单版本；随后优先级演进把 `manifest_version` 单调推进到 3，并触发陈旧防护。v1/v2 的最终版本包含升级和修复导致的版本推进。

v3 首次运行的预算为 `max_parallel=3`、`per_node_kb=3`。计划分两轮：

```text
wave 1: rep-0001, rep-0002, rep-0003
wave 2: rep-0004
queue : rep-0004 / v3-d / 3 KB / per_node_quota
```

目标节点 `v3-d` 在第一轮已使用 2 KB，再装入 3 KB 会超过 3 KB 配额，因此任务被排到下一轮；执行后所有副本摘要与真值一致。

## 陈旧防护实测

首次 v3 修复后把 `v3-c` 优先级从 2 改为 6：

```text
previous manifest_version = 2
current  manifest_version = 3
version_incremented = true
affected_nodes = [v3-c]
affected_blocks = [blk-0001, blk-0002, blk-0003]
ledger recovery = content_digest
current view during recovery = null
```

随后重新运行完整链路，当前视图整体替换为新的 before/after 快照，重放通过。

## 异常注入实测

- 块文件截断：退出码 3，`E_BLOCK`，实测路径 `$.nodes[v3-a].blocks[blk-0001].size_bytes`，期望 4096 / 实际 4095。
- 同尺寸内容翻转：退出码 4，`E_INTEGRITY`，实测路径 `$.block[blk-0001].digest`，错误同时给出节点和期望/实际摘要。
- `max_parallel=0`：退出码 5，`E_BUDGET`，字段路径 `$.budget.max_parallel`。
- 单块修复量大于 `per_node_kb`：`E_BUDGET`，包含块、目标节点、所需 KB 和配额。

`view` 在当前无 Tk 的环境中按设计返回退出码 9 和 `E_ENV`，核心链路不受影响。

## 独立报告与重放

```text
report: exit 0, current.phase=after, current.manifest_version=3,
        reopen_check.replaced=false, stale.affected_nodes=[v3-c]
replay: exit 0, distribution_checks=12, truth_checks=3,
        recovery.replaced=false
```

## 强制终止实测

- 升级：`upgrade --crash-after 1` 以 86 退出；检查点有 1 条已提交记录，原清单仍是 v1，备份只读；删除已提交节点的块文件后续做成功，证明该记录未被再次读取或升级。
- 修复：`run --crash-after-task 1` 以 86 退出；重新打开时旧视图置空、未完成任务为 `unknown`；再次运行时先对账已落盘副本，再完成余下任务并生成新的前后快照。

## 文件产物

当前工作区保留最近一次 demo 的 `data/groups/*.json`、只读备份、升级检查点、`nodes/*` 块文件、`ledger/*.sqlite3` 和 `data/demo_report.json`，可直接用于查看；重新执行 `demo` 会按已知样本路径重建这些产物。
