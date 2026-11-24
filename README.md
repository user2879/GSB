# 定点 TLS 证书链与确定性握手领域组件

本仓库从空目录首次落地一个可集成的本地 TLS 握手/证书链验证组件。它用真实 `127.0.0.1` 回环套接字驱动确定性握手机制，用纯整数定点完成证书签名、有效期、路径长度和信任锚验证，并用 SQLite 账本保存逐批中间量、误差界、判定与恢复游标。

组件没有使用真实证书、真实 CA、外部账号或容器。样本中的签名算法是用于教学/验收的确定性 toy-RSA 校验，不用于生产 TLS。

## 交付内容

- 两份强校验 JSON：证书链声明、握手脚本。
- 回环子进程服务与父进程真实套接字协议。
- 整数定点证书链验证、误差三段归因和稳定决胜。
- 证书/消息依赖失效传播与增量缓存。
- SQLite 批次事务、内容摘要、整体快照替换。
- 优雅信号退出报告、强制终止账本恢复、逐消息重放校验。
- Tkinter 诊断界面及可测试的无界面诊断模型。
- 固定种子样本、异常注入、端到端演示和 40 项自动化测试。

## 快速开始

### 1. Windows 环境准备

```powershell
python -V
python -m pip install numpy==1.26.4
python -c "import numpy; print(numpy.__version__)"
python -c "import tkinter, sqlite3, socket, subprocess; print('stdlib imports ok')"
python tools/check_environment.py
netstat -ano | findstr ":23771"
```

要求 Python 3.11、`numpy 1.26.4`、可导入的 `tkinter`，且默认端口 `23771` 未被占用。核心计算不依赖浮点或 numpy；numpy 按任务环境要求安装并用于环境自检。

### 2. 生成固定种子样本

```powershell
python tools/generate_samples.py
```

样本文件：

| 文件 | 用途 |
| --- | --- |
| `samples/certificates.json` | 9 张证书、固定点参数、信任锚 |
| `samples/handshake.json` | 8 条客户端/服务端消息 |
| `samples/certificates_modified.json` | 增量演示：重签并更新 `leaf-expired` 有效期 |
| `samples/certificates_self_cycle.json` | 自环证书声明 |
| `samples/certificates_bad_algorithm.json` | 不支持算法 |
| `samples/handshake_bad_seq.json` | 不连续消息序号 |
| `samples/corrupted.json` | 损坏 JSON |

### 3. 验证与运行

```powershell
python -m tlschain.cli validate --certs samples/certificates.json --handshake samples/handshake.json
python -m tlschain.cli run --certs samples/certificates.json --handshake samples/handshake.json --ledger artifacts/full/ledger.sqlite --port 23771 --reset
python -m tlschain.cli recover --certs samples/certificates.json --handshake samples/handshake.json --ledger artifacts/full/ledger.sqlite --port 23772
```

预期判定：`PASS=5`、`FAIL=1`、`INDETERMINATE=2`，4 个提交批次，游标依次为 `2,4,6,8`，所有消息与脚本期望一致。

### 4. 增量重算

```powershell
python -m tlschain.cli incremental --base-certs samples/certificates.json --base-handshake samples/handshake.json --changed-certs samples/certificates_modified.json --changed-handshake samples/handshake.json
```

预期仅失效 `leaf-expired` 与消息 `4..8`；证书与消息 `1..3` 的中间量/结论逐字节保留；增量结果与空白全量重建的所有验证字段一致。

### 5. 两类终止

```powershell
python -m tlschain.cli run --certs samples/certificates.json --handshake samples/handshake.json --ledger artifacts/graceful/ledger.sqlite --port 23773 --reset --terminate-after 4 --termination graceful
python -m tlschain.cli recover --certs samples/certificates.json --handshake samples/handshake.json --ledger artifacts/graceful/ledger.sqlite --port 23774

python -m tlschain.cli run --certs samples/certificates.json --handshake samples/handshake.json --ledger artifacts/hard/ledger.sqlite --port 23775 --reset --terminate-after 4 --termination hard
python -m tlschain.cli recover --certs samples/certificates.json --handshake samples/handshake.json --ledger artifacts/hard/ledger.sqlite --port 23776
```

优雅路径通过已安装信号处理器停止并写入完整 `exit-report.json`。强制路径无退出报告，只有已提交账本；恢复时消息 `5..8` 先标为未知并要求重跑，已提交前缀 `1..4` 逐字段重放校验。

### 6. 异常注入

```powershell
python -m tlschain.cli validate --certs samples/corrupted.json --handshake samples/handshake.json
python -m tlschain.cli validate --certs samples/certificates.json --handshake samples/handshake.json --anomaly self-cycle
python -m tlschain.cli run --certs samples/certificates.json --handshake samples/handshake.json --ledger artifacts/timeout.sqlite --port 23777 --drop-client-seq 3
python -m tlschain.cli run --certs samples/certificates.json --handshake samples/handshake.json --ledger artifacts/reply.sqlite --port 23778 --omit-reply-field seq
```

分别得到带证书/字段定位的 `E_SPEC`，以及定位到消息序号的 `E_TIMEOUT`、`E_REPLY`。端口被占用时服务子进程返回 `E_PORT`。

### 7. 诊断界面与一键演示

```powershell
python -m tlschain.cli diagnostics --certs samples/certificates.json --handshake samples/handshake.json --ledger artifacts/full/ledger.sqlite
python -m tlschain.cli gui --certs samples/certificates.json --handshake samples/handshake.json --ledger artifacts/full/ledger.sqlite
python tools/demo.py
python -m unittest discover -v
```

`gui` 以只读方式打开账本，刷新只重建视图模型，不写账本。`demo.py` 生成 `artifacts/demo/demo-summary.json`，覆盖完整运行、增量、两类终止和恢复。

## 目录

| 路径 | 说明 |
| --- | --- |
| `tlschain/spec.py` | JSON 加载、强校验、E_SPEC 定位 |
| `tlschain/fixed.py` | 整数定点验证、误差界、稳定裁决 |
| `tlschain/session.py` | 全量/增量会话与失效传播 |
| `tlschain/service.py` | 回环子进程服务 |
| `tlschain/service_client.py` | 父进程真实套接字客户端 |
| `tlschain/runner.py` | 批次提交、信号终止、恢复重放 |
| `tlschain/ledger.py` | SQLite 账本、摘要、快照替换 |
| `tlschain/gui_model.py` | 稳定只读诊断模型 |
| `tlschain/gui.py` | Tkinter 界面 |
| `tools/generate_samples.py` | 固定种子样本生成 |
| `tools/demo.py` | 完整可重复验收演示 |
| `tests/` | 40 项单元/集成测试 |
| `docs/TECHNICAL.md` | 协议、定点公式、账本与恢复细节 |
| `docs/VERIFICATION.md` | 本次实际执行结果与环境记录 |

## 关键约束

- 端口、会话地址固定为 `127.0.0.1`，默认高位端口 `23771`；不访问外部目标。
- 所有时间、模数、签名中间量、路径权重均为整数；除法统一向负无穷截断，溢出向有符号 `max_bits` 范围饱和，饱和会显式进入不可区分状态。
- 账本每个消息批次在单个 SQLite 事务中提交；强制终止不会伪造退出报告。
- 快照格式版本或内容摘要不一致时整体替换，不做增量合并。
- toy 签名算法是确定性验证模型，不声明提供真实密码学安全性。

## 实测摘要

在当前工作区实际执行 `python -m unittest discover -v`：40 项全部通过。`python tools/demo.py` 得到：

- 完整运行：`COMPLETE`，游标 8，批次 `[2,4,6,8]`，`PASS/FAIL/INDETERMINATE=5/1/2`。
- 增量：仅失效 `leaf-expired` 和消息 `4..8`，增量与全量验证字段一致，未失效证书/消息逐字节一致。
- 优雅终止：游标 4，`SIGINT`，退出报告存在，恢复后未知消息 `5..8` 被重跑。
- 强制终止：返回 `SIGKILL` 终止码，退出报告与运行报告均不存在，账本仅含批次 1–2；恢复后游标 8。
- 诊断刷新：连续两次输出逐字节一致。

详细环境和完整结果见 `docs/VERIFICATION.md`。
