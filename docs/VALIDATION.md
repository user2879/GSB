# 实际验证记录

验证日期：2026-11-20（Asia/Shanghai）。未使用 Docker，未访问外部站点；网络交互只发生在 `127.0.0.1` 回环端口。

## 环境

- Python：`3.11.0rc1`
- 虚拟环境：项目根目录 `.venv`
- numpy：`2.4.6`
- pytest：`9.2.0`
- 当前 Linux 容器的系统 Tkinter：缺失（`No module named tkinter`）
- UI 无界面模型：已验证；Windows 官方 Python 自带 Tkinter，可用 `ui` 命令启动窗口

虚拟环境最初缺少 `ensurepip`，使用官方 `get-pip.py` 引导后安装了 numpy、pytest 与可编辑项目包。

## 自动化测试

命令：

```bash
.venv/bin/python -m pytest -q
```

结果：`28 passed`（28 个测试，0 失败）。测试分布：

| 文件 | 数量 | 覆盖 |
| --- | ---: | --- |
| `test_config.py` | 5 | 整体拒绝、字段路径、延迟/可达性、E_SCRIPT、重复键 |
| `test_congestion.py` | 4 | 定点 RTT/RTO、慢启动、拥塞避免、快速重传、超时、Karn、迁移日志 |
| `test_incremental.py` | 3 | 每事件增量=全量、未受影响行逐字节不变、依据/版本/依赖 |
| `test_versions.py` | 3 | 整集替换、旧结论 stale 只读、版本单调与禁止隐式合并 |
| `test_snapshot.py` | 4 | 快照保存/校验、历史版本、整集恢复、缺版本、首差异 E_REPLAY |
| `test_recovery.py` | 2 | 真实子进程 `os._exit(23)` 强制终止恢复、正常恢复 |
| `test_peer_loopback.py` | 3 | 回环 ACK/重复 ACK/丢包/延迟/乱序、E_PORT、E_CONNECT |
| `test_cli_ui.py` | 4 | 无界面稳定渲染、E_SCRIPT/E_PORT/E_VERSION 结构化日志 |

## 一键验收脚本

命令：

```bash
.venv/bin/python scripts/verify_acceptance.py
```

实测输出：

```json
{
  "event_script": {
    "archived_fields": 10,
    "baseline_snapshot_id": 1,
    "events_ingested": 20,
    "evolved_to": "C2/S2",
    "historical_snapshot_ids": [2, 3],
    "incremental_equals_full": true,
    "replacement_fields": 3,
    "snapshots_verified": 3
  },
  "loopback_peer": {
    "dropped_seqs": [6001],
    "event_count": 19,
    "final_state": "CLOSED",
    "incremental_equals_full": true,
    "peer": "127.0.0.1:<ephemeral-high-port>",
    "reordered_responses": 1,
    "responses": 6,
    "rounds": 2
  }
}
```

说明：回环端口每次由操作系统分配，实测值在 `47000` 左右的高位端口；上表以占位符记录非本质随机值。

## 关键验收结论

1. 20 条事件逐条到达时，每一步的十项增量结论与从零全量重放的 `value_json`、`revision`、`basis_event_seq`、依赖列表及配置/脚本版本完全一致。
2. 在一条不影响连接状态的 `SEND` 事件前后，`conclusions.connection_state` 整行字段逐字节相同。
3. 从配置/脚本 `C1/S1` 演进到 `C2/S2` 后，当前 10 项结论全部为 `C2/S2`；旧 10 项进入 `conclusion_history` 且 `stale=1`，不参与当前 UI/汇总。
4. 子进程在事件以 pending 落库后直接 `os._exit(23)`。重开账本得到 `unknown_event_seqs=[1]`、`rerun_required=true`、`exit_report_available=false`；执行 `retry` 后按持久化载荷重新推导并恢复一致性。
5. 篡改快照事件 5 的 RTT 样本后，校验以 `E_REPLAY` 报出 `first_event_seq=5` 和首个字段差异；错误 context 同时带 expected/actual。
6. 快照 `config_version=NULL` 时以 `E_VERSION` 拒绝，字段包含 `snapshot.config_version`。
7. 真实回环会话产生 ACK、重复 ACK、6001 丢包、延迟和 1 个乱序应答；最终状态 CLOSED，增量=全量。
8. `inject script`、`inject port`、`inject version` 分别把 `E_SCRIPT`、`E_PORT`、`E_VERSION` 写入 JSON-lines 日志，CLI 返回码为 2。

## Tkinter 说明

当前验证容器没有系统 Tkinter，因此没有伪造“窗口已打开”的结论。无界面视图已通过 `ui --headless` 和测试验证：固定十项顺序、指标头部、受影响标记、依赖事件、版本、快照列表和结构化日志均稳定。Windows 标准 Python 环境可按 README 直接启动 Tkinter。

