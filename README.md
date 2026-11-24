# Batch Attribution Diagnostic Component（本地推理批归属诊断组件）

一个完全离线、确定性的 Python 包：把本地推理请求按批切分并分配到若干
`127.0.0.1` 回环工作器，工作器用 NumPy 执行确定性前向计算并回传带批次归属的
回执；组件把 r1/r2/r3 三种历史回执归一为同一内部视图，并沿
「工作器参数版本 → 批次 → 请求归属」依赖链做失效传播、仅失效子集重算、
SQLite 账本持久化、恢复重放与独立验证。

- Python ≥ 3.11，仅依赖 `numpy`；Tkinter 使用 Windows 官方安装包内置版本。
- 不使用 Docker，不访问任何外部服务或在线资源。
- 所有回环端口均来自命令行声明，运行前做可绑定检查，互不重复。

## 目录结构

```
batch_diagnostic/
  errors.py          业务错误码 E_SPEC/E_FIELD/E_CHAIN/E_REPLAY/E_SNAPSHOT/E_VERIFY
  canonical.py       显式稳定序列化与 SHA-256 摘要（与字典/派发顺序无关）
  logging_utils.py   结构化 JSONL 日志
  spec.py            请求流/工作器/预算 JSON 强校验与版本指纹
  receipt.py         r1/r2/r3 编解码、E_FIELD 校验、带补齐依据的归一视图
  planning.py        确定性批计划、批标识推导、显式余数批规则
  inference.py       NumPy 确定性前向与顺序无关输出摘要
  invalidation.py    参数版本→批次→请求的失效传播与逐步依据
  network.py         真实回环套接字帧、真实超时、幂等重试
  scheduler.py       线程池 + 全局/每工作器并发信号量、单调时钟排队计时
  engine.py          工作器服务、批执行、本地复核、逐批账本提交
  ledger.py          SQLite 单事务提交、内容摘要、整体替换恢复、只读备份
  replay.py          恢复一致性重放（E_REPLAY，定位首个分歧批/请求/字段）
  verifier.py        只读账本+原始声明的独立验证（E_VERIFY/E_CHAIN）
  gui.py             Tkinter 诊断界面（无显示环境可导入/降级为 JSON）
  cli.py             命令行入口 batch-diagnostic
samples/             确定性本地样本与历史回执、样本生成器
scripts/             check_ports.py 端口预检、demo.py 可重复演示
tests/               pytest 套件（36 个用例）
docs/VALIDATION.md   实际验证记录
```

## 安装

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
# 可选：以可编辑方式安装并获得 batch-diagnostic 命令
pip install -e .
```

## 快速开始

```bash
# 0) 生成/刷新确定性样本（已随仓库提供）
python samples/generate_samples.py

# 1) 回环端口预检（必须全部可绑定且互不重复）
python scripts/check_ports.py \
  --requests samples/requests.json \
  --workers samples/workers.json \
  --budget samples/budget.json

# 2) 一条命令跑完可重复演示（归一化→计划→回环推理→失效重算→重放→注入）
python scripts/demo.py
```

等价的模块化命令：

```bash
python -m batch_diagnostic.cli check-ports --requests ... --workers ... --budget ... --ledger run/d.sqlite
python -m batch_diagnostic.cli plan        --requests ... --workers ... --budget ...
python -m batch_diagnostic.cli run         --requests ... --workers ... --budget ... --ledger run/d.sqlite
python -m batch_diagnostic.cli rerun       --requests ... --workers ... --budget ... --ledger run/d.sqlite --new-workers samples/workers_v2.json
python -m batch_diagnostic.cli replay      --requests ... --workers ... --budget ... --ledger run/d.sqlite
python -m batch_diagnostic.cli verify      --requests ... --workers ... --budget ... --ledger run/d.sqlite
python -m batch_diagnostic.cli inject      --kind request_corrupt|dangling_ref|snapshot_digest ...
python -m batch_diagnostic.cli dashboard   --requests ... --workers ... --budget ... --text
```

## 关键设计约定

- 稳定序列化：所有标识/摘要基于键排序、无空白的 canonical JSON + SHA-256，
  任意字典遍历顺序与派发顺序下批划分和批标识逐字节一致。
- 批标识：`batch-<worker_id>-<batch_index:04d>-<digest(worker_id,batch_index,arrival_seqs)>`。
- 余数批：剩余不足 `batch_size` 时仍生成一个尾随部分批，`remainder=true` 并附带
  显式依据，不丢请求、不改顺序。
- 归一补齐：r1/r2 缺失字段使用显式默认值（如 `param_version=1`、
  `elapsed_ms=0`、`worker_id="unknown"`、`rank=0`、`duplicate_ref=null`、
  `param_digest="unavailable"`），每个补齐字段记录默认值与可回溯依据。
- 顺序无关摘要：批输出摘要在哈希前对请求标识排序，处理顺序变化不改变摘要。
- 失效语义：种子/维度/参数版本/工作器集合/批大小变化 → 失效该工作器全部批
  → 失效这些批上的请求；仅重算失效集合，未受影响批的回执、归属与批版本号
  逐字节保持不变（演示与测试逐项断言）。
- 账本：每批在单事务内写归属、回执、失效轨迹、预算消耗与进度游标；头部携带
  整数格式版本与内容摘要。重开先比对工作器集合版本/批大小版本/内容摘要，
  不一致则保留只读备份并整体替换；未完整提交的批标记为 unknown 要求重跑。
- 并发：真实线程池 + 全局信号量 + 每工作器信号量，记录单调时钟
  `wait_started_ms/wait_ended_ms`；套接字设置真实超时，超时按重试上限做
  幂等重试，不改变单次处理语义。

## 业务码

| 码 | 触发场景 |
| --- | --- |
| `E_SPEC` | 请求流/工作器/预算声明非法，定位首个非法条目与字段 |
| `E_FIELD` | 回执未知版本、缺字段、负耗时、名次越界等 |
| `E_CHAIN` | 重复投递引用悬空等归属链断裂 |
| `E_REPLAY` | 恢复重放与账本首个分歧批/请求/字段 |
| `E_SNAPSHOT` | 快照工作器集合/批大小版本或内容摘要不符 |
| `E_VERIFY` | 独立验证工具发现首个违规批 |

## 测试

```bash
python -m pytest
```
