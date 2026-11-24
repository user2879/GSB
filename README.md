# batchdiag — 本地推理批归属诊断

确定性 Python 3.11 包：把 r1/r2/r3 三种归属回执归一为可复核 SQLite 账本，
沿 `工作器参数版本 → 批次 → 请求归属` 依赖链判定失效边界，并保证增量重算
与空白全量重建逐字节一致。纯本地样本 + 127.0.0.1 真实 TCP 工作器子进程，
无容器、无外部服务。

## 快速开始

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
bash scripts/demo.sh                                     # 端到端演示
.venv/bin/python -m unittest discover -s tests -v        # 31 个测试
```

Windows（PowerShell）等价流程见 `scripts/demo.ps1`；激活脚本为
`.venv\Scripts\Activate.ps1`。

## 常用命令

```bash
PY=.venv/bin/python
$PY -m batchdiag.cli gen-samples --samples-dir samples
$PY -m batchdiag.cli validate   --requests samples/requests.json --workers samples/workers.json --rules samples/merge_rules.json
$PY -m batchdiag.cli normalize  --requests samples/requests.json --workers samples/workers.json --rules samples/merge_rules.json --receipts samples/receipts_r3.json
$PY -m batchdiag.cli plan       --requests samples/requests.json --workers samples/workers.json --rules samples/merge_rules.json
$PY -m batchdiag.cli run        --requests samples/requests.json --workers samples/workers.json --rules samples/merge_rules.json --ledger artifacts/ledger.sqlite3
$PY -m batchdiag.cli incremental --requests samples/requests.json --workers samples/workers.json --new-workers samples/workers_v2.json --rules samples/merge_rules.json --ledger artifacts/inc.sqlite3
$PY -m batchdiag.cli verify     --requests samples/requests.json --workers samples/workers.json --rules samples/merge_rules.json --ledger-a artifacts/a.sqlite3 --ledger-b artifacts/b.sqlite3
$PY -m batchdiag.cli scheduler  --requests samples/requests.json --workers samples/workers.json --rules samples/merge_rules.json [--timeout-demo]
$PY -m batchdiag.cli snapshot   ... --ledger run/snap.sqlite3 [--tamper]
$PY -m batchdiag.cli crash      ... --ledger run/crash.sqlite3 --after 2
$PY -m batchdiag.cli recover    ... --ledger run/crash.sqlite3
$PY -m batchdiag.cli replay     ... --ledger run/crash.sqlite3
```

## 目录

- `batchdiag/`：可导入包（规格、回执编解码、切批、推理、失效、调度、账本、引擎、CLI、可选 GUI）
- `samples/`：固定整数种子生成的请求流、工作器、规则、r1/r2/r3 回执与非法样本
- `scripts/`：环境准备、演示、测试脚本（bash + PowerShell）
- `tests/`：31 个 unittest，覆盖全部 8 项业务要求与错误码
- `docs/DESIGN.md`：架构、稳定序列化规则、依赖链与错误码设计
- `docs/VALIDATION.md`：实际验证结果与环境限制

错误码：`E_SPEC`（规格）、`E_FIELD`（回执字段）、`E_CHAIN`（引用链路）、
`E_TIMEOUT`（套接字超时）、`E_REPLAY`（重放分歧）、`E_SNAPSHOT`（快照）。
