# 技术说明

## 运行

```bash
python3 -m pip install numpy pytest
python3 -m topic_ledger.cli generate
python3 -m topic_ledger.cli validate
python3 -m pytest -q
python3 -m topic_ledger.cli demo
```

一键演示会启动真实的 `127.0.0.1` JSONL Worker 子进程，重放 `config/commands.json`，
执行租约到期回收探针、共享词项账本闭合检查、增量/空白重放比对和最终全量重建比对，
并写出：

- `artifacts/verification-run.json`：机器可读的完整判定；
- `artifacts/verification-summary.md`：简要结果；
- `artifacts/pytest-output.txt`：测试输出。

## 模块边界

- `topic_ledger/validation.py`：节点声明、推断配置、文档流和命令流的首个越界字段校验。
- `topic_ledger/samplegen.py`：固定整数种子生成受控词表、文档流、节点声明和示例命令。
- `topic_ledger/sampling.py`：NumPy 定点整数权重采样、轮次指纹、软分配摘要和逐词项计数。
- `topic_ledger/worker.py`：子进程入口与回环 TCP JSONL 协议，维护节点并发闸和重复投递摘要闸。
- `topic_ledger/cluster.py`：子进程拉起、JSONL 客户端、空闲端口辅助和进程清理。
- `topic_ledger/leases.py`：单调代次租约、到期回收、幂等在途槽和全局/节点并发池。
- `topic_ledger/scheduler.py`：采样点数预算、节点租约上限、稳定分片顺序和待办优先级。
- `topic_ledger/ledger.py`：权威来源排序、被覆盖来源账本、全局分布、逐主题闭合和冻结。
- `topic_ledger/cache.py`：内容寻址缓存与临时目录写入、版本目录原子重命名、当前指针原子替换。
- `topic_ledger/engine.py`：命令处理、失效判定、并发调度、快照恢复和进度摘要。
- `topic_ledger/replay.py`：逐字段深度差异、同命令流重放和最终集合全量重建一致性判定。
- `topic_ledger/demo.py`：验收场景聚合与实际运行结果落盘。

## 确定性口径

所有语义指纹均使用 `sort_keys=True`、紧凑分隔符和 UTF-8 的 canonical JSON，再计算 SHA-256。
缓存键由文档内容指纹、词项计数切片、主题参数指纹、定点位宽档位、采样轮数和每轮中间量派生。

Worker 另外返回真实 `time.monotonic_ns()` 测得的 `elapsed_monotonic_ms`，用于性能观测；
权威裁决使用回执中的 `monotonic_ms` 逻辑时钟字段。默认引擎使用固定起点的 `ManualClock`，
因此命令流、租约代次、到期顺序、权威来源和业务码计数可跨进程/跨并发度逐字段复核。
真实 TTL 回收仍由 `LeaseManager(clock=monotonic_ms)` 与实际 `sleep` 探针验证；等待只用于越过 TTL，
配额释放由 `SlotTicket.release()` 真实完成。

## 权威来源与闭合

同一 `term_id` 的多个分片来源按以下键排序：

1. `topic_parameter_version` 降序；
2. `monotonic_ms` 降序；
3. `node_id` 字典序升序；
4. 为保证全序追加 `shard_id` 与 `receipt_digest` 作为稳定末级裁决。

首位为权威来源，其余来源连同覆盖来源写入 `overwrites`。全局分布只累加权威来源，
按 `term_id`、`topic_id` 双升序输出整数计数。每个主题比较：

```text
sum(global_distribution[*][topic])
  == sum(authoritative_term_allocation[topic])
```

不一致时输出 `E_LEDGER_MISMATCH`，冻结对应主题；后续触碰该主题的回执以 `E_LEDGER_FROZEN` 拒绝。

## 一致性判定范围

`replay_equivalence()` 将同一最终命令流分别交给两个空白状态目录，以不同并发度运行，
再对完整 canonical report 做深度差异比较。比较内容包括分片回执、逐词项计数、权威/覆盖依据、
全局分布、闭合状态、待办顺序、失效复用清单、租约终结记录和业务码计数。

`final_full_rebuild_equivalence()` 从增量运行后的最终文档集合与配置出发，在空白目录强制全量采样，
比较文档、分片回执、权威来源、覆盖账本、全局分布和闭合字段。运行时遥测（真实耗时、PID、端口、
状态目录）不进入语义报告。

## 快照恢复

每次命令处理后，快照状态先完整写入版本目录下的 `.tmp-*` 临时目录，fsync 后原子重命名为
不可变版本目录，再通过临时指针文件加 `os.replace()` 切换 `current.json`。恢复时比较当前配置哈希：

- 哈希相同：`resume`，恢复待办、已消耗预算、业务码计数、缓存清单和回执；
- 哈希不同：`rebuild`，主题维度等不兼容状态重建，内容寻址缓存中键相同的未受影响产物仍可命中。
