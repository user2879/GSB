# QoS 令牌桶链诊断组件

面向本地 QoS 整形平台的确定性诊断、失效重算、TCP 整形器编排、SQLite 账本恢复与独立校验实现。无需 Docker 或外部网络；所有整形器仅监听 `127.0.0.1`。

原始任务书保留在 `docs/task-spec.md`，实现与验证细节见 `docs/TECHNICAL.md`、`docs/VERIFICATION.md`。

## 环境

- Python 3.11+，Windows、Linux 或 WSL 均可。
- 唯一第三方依赖：`numpy==1.26.4`，仅用于固定种子生成样本。

```powershell
python -m pip install -r requirements.txt
```

## 一键验证

Windows PowerShell：

```powershell
.\run_validation.ps1
```

Windows CMD：

```cmd
run_validation.cmd
```

跨平台手动执行：

```bash
python -m qos_diag sample --workspace .
python -m qos_diag drill --workspace . --output artifacts/reports/acceptance-drill.json
python -m unittest discover -s tests -v
python -m qos_diag verify --spec-dir samples/edited_param --ledger artifacts/ledger/qos.db
```

`drill` 会真实启动 9 个回环 TCP 整形器子进程，演练：

- 三级令牌桶判定与 event/window 两条等价路径；
- 强制 `kill -9`/`TerminateProcess` 风格杀死 `sh-02-relay`；
- 未提交批次标记 `unknown`，重启后从 SQLite 已提交游标恢复；
- 速率、突发、到达毫秒编辑后的链/窗口失效传播与子集重算；
- 账本格式版本/内容摘要不一致时整体隔离替换；
- 真实 socket 超时探针与全局/管道并发配额探针；
- 生成 `artifacts/reports/acceptance-drill.json`。

## 常用命令

```bash
# 强校验样本；失败输出 E_SPEC、object_id、field
python -m qos_diag validate --spec-dir samples/base

# 只重放确定性引擎，不启动子进程
python -m qos_diag diagnose --spec-dir samples/base --output artifacts/snapshots/base.json
python -m qos_diag diagnose --spec-dir samples/base --path event
python -m qos_diag diagnose --spec-dir samples/base --path window

# 真实 TCP 子进程 + 账本运行
python -m qos_diag clean --workspace .
python -m qos_diag run --spec-dir samples/base --ledger artifacts/ledger/qos.db \
  --snapshot artifacts/snapshots/base.json

# 强杀子进程后恢复
python -m qos_diag recover --spec-dir samples/base --ledger artifacts/ledger/qos.db

# 编辑后版本化增量重算
python -m qos_diag edit --old-dir samples/base --new-dir samples/edited_param \
  --ledger artifacts/ledger/qos.db \
  --invalidation-out artifacts/snapshots/invalidation_param.json
python -m qos_diag verify --spec-dir samples/edited_param --ledger artifacts/ledger/qos.db
```

校验失败返回 `E_VERIFY`，JSON 中包含首个不一致的 `event_seq`、`pipeline_id`、`stage`、`field`、期望值与账本值。

## 目录

- `qos_diag/model.py`：JSON schema 与 `E_SPEC` 强校验。
- `qos_diag/fixedpoint.py`：`Q=10^scale` 定点整数、饱和规则、误差分类。
- `qos_diag/engine.py`：三级串联桶、逐事件路径、同类滑窗批量路径。
- `qos_diag/diagnosis.py`：路径选择、窗口/逐级统计、误差界。
- `qos_diag/invalidation.py`：链级、时间后缀、重叠窗口失效闭包与增量复用。
- `qos_diag/runtime.py`：并发治理、真实 socket 客户端、子进程生命周期。
- `qos_diag/shaper_server.py`：每个整形器声明对应一个 `127.0.0.1:port` 子进程。
- `qos_diag/ledger.py`：SQLite 原子事务、版本/摘要、批次与游标。
- `qos_diag/verify.py`：只读原始 JSON + 账本的独立重放校验。
- `qos_diag/drill.py`：端到端验收演练。
- `samples/`：固定种子 `287` 生成的 base/rate/burst/arrival 样本。
- `tests/`：12 个核心与真实运行时/账本测试。

## 关键稳定性约束

- 所有哈希与字节比较使用 `sort_keys=True`、紧凑分隔符的 canonical JSON。
- 事件按输入序号推进；管道、整形器、窗口和失效集合按 ID/时间稳定排序。
- 分批只是连续 fold 的边界，不改变状态，因此任意批次大小输出逐字节一致。
- 未受影响的 stage record 直接复用；测试与演练会逐项比较 canonical bytes。
- socket 超时只来自 `socket.create_connection(... timeout=...)` 与 `sock.settimeout(...)`，没有 `future.wait_for` 等模拟超时。
