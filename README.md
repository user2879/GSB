# 本地推理批归属诊断

这是一个从零实现、可重复验证的 Python 包。它把本地推理请求切成确定性批次，通过
`127.0.0.1` 回环套接字工作器子进程执行 NumPy 前向计算，将 r1/r2/r3 三代回执归一为
可复核账本，并沿「工作器参数版本 → 批次 → 请求归属」传播失效。账本使用 SQLite 单批
事务提交，支持强制终止后的恢复、会话快照整体替换和重放差异定位。

项目不依赖 Docker、外部服务、账号或业务网络。工作器只绑定字面量 `127.0.0.1`。

## 快速开始

要求 Python 3.11+、NumPy 1.26.4。tkinter 仅用于环境确认，业务链路不依赖 GUI。

```bash
python3 --version
python3 -m pip install numpy==1.26.4
python3 -m pip install . --no-deps --no-build-isolation

# 也可以不安装，直接从源码运行
export PYTHONPATH="$PWD/src"
```

运行本地固定种子样本、完整验收演示和测试：

```bash
PYTHONPATH=src python3 scripts/run_demo.py
PYTHONPATH=src python3 -m pytest -q
```

演示结果写入 `artifacts/demo/summary.json`，本次实际运行记录见
[`docs/RUN_LOG.md`](docs/RUN_LOG.md)，详细规则见 [`docs/DESIGN.md`](docs/DESIGN.md)。

## 目录

```text
src/inference_attribution/
  spec.py          请求、工作器、归并规则强校验
  model.py         PCG64 参数生成、NumPy 前向和摘要
  planner.py       稳定批划分与批标识
  receipts.py      r1/r2/r3 编解码、补齐、归一
  worker_server.py 回环套接字工作器子进程
  workers.py       子进程生命周期、协议客户端、输出复核
  scheduler.py     全局/工作器配额与单调时钟排队记录
  ledger.py        SQLite 事务账本、恢复状态、完整性摘要
  invalidation.py  失效传播、重放差异定位
  snapshot.py      会话快照、整体替换恢复
  engine.py        逐批派发、回执回收和账本提交
  cli.py           可执行命令行
samples/            固定整数种子生成的请求、声明、规则和三代回执
scripts/            样本重建与端到端演示
tests/              规格、格式、计划、调度、失效、强杀恢复、快照和重放测试
```

## 样本与确定性

- 请求输入由整数种子 `90125` 生成；两个工作器参数种子分别为 `41001`、`41002`。
- 样本含 6 个请求、2 个工作器、3 个批次；r1、r2、r3 各覆盖一个批次。
- 批标识由 `worker_id`、`batch_seq`、参与请求的 `arrival_seq` 列表经排序键、紧凑 JSON
  和 SHA-256 推导，不含字典遍历顺序。
- 每个请求输出摘要和批次输出摘要均绑定请求标识集合、参数版本序号与参数摘要；批内
  处理顺序不进入摘要。

样本可重建：

```bash
PYTHONPATH=src python3 scripts/build_samples.py
```

## 常用 CLI

```bash
# 强校验
inference-attribution validate \
  --requests samples/requests.json \
  --workers samples/workers.json \
  --rules samples/merge_rules.json

# 三代回执归一
inference-attribution normalize \
  --requests samples/requests.json \
  --workers samples/workers.json \
  --rules samples/merge_rules.json \
  --r1 samples/receipts_r1.jsonl \
  --r2 samples/receipts_r2.jsonl \
  --r3 samples/receipts_r3.jsonl \
  --output artifacts/normalized.json

# 真实回环工作器运行；--crash-after 1 用于强杀演练
inference-attribution run \
  --requests samples/requests.json \
  --workers samples/workers.json \
  --rules samples/merge_rules.json \
  --ledger artifacts/full.sqlite

# 读取强杀后账本，不会伪造退出报告
inference-attribution recover --ledger artifacts/crash.sqlite

# 会话快照与整体替换恢复
inference-attribution snapshot \
  --workers samples/workers.json \
  --ledger artifacts/full.sqlite \
  --output artifacts/snapshot.json
inference-attribution restore \
  --workers samples/workers_param_update.json \
  --snapshot artifacts/snapshot.json \
  --ledger artifacts/restored.sqlite
```

错误统一输出为包含 `code/object/field/line/message` 的 JSON 诊断。

## 错误码

| 错误码 | 含义 |
| --- | --- |
| `E_SPEC` | 请求流、工作器声明或归并规则非法；指出首个非法对象和字段 |
| `E_FIELD` | 回执字段缺失、类型/范围非法、名次重复或越界；带行号 |
| `E_REF` | 回执中的批次、请求或重复投递引用不存在或不匹配；带行号 |
| `E_PORT` | 工作器端口已被占用或工作器未能开始监听 |
| `E_PROTO` | 回环协议响应缺字段、字段非法或输出摘要不一致 |
| `E_LEDGER` | SQLite 账本格式版本或内容摘要不完整（内部一致性错误） |
| `E_SNAPSHOT` | 快照内容摘要、账本摘要、游标或格式版本不一致 |
| `E_REPLAY` | 重放结果与参考账本不一致；定位首个批标识和字段 |

## 保证与边界

- 同一规格文件的任意数组顺序会得到逐字节一致的稳定批计划。
- 未失效批的账本行不会被重新写入；测试逐字节比较该行。
- 工作器耗时是真实观测值，因此独立两次运行可能有不同毫秒数。归属结论、批次摘要、
  参数绑定和重放比较不使用该易变耗时作为语义字段。
- 强制终止只会留下 `pending` 未知批；只有完整提交的批可恢复。程序明确声明无法生成
  被强杀进程的可信退出报告。
