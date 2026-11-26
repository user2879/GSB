# 本地确定性 TCP 状态观测交付说明

## 一键复现

在 WSL Ubuntu 项目根目录执行：

```bash
./scripts/reproduce.sh
```

脚本会：

1. 在 `.venv` 中准备 Python 3.11 与 `numpy==1.26.4`；
2. 以可编辑方式安装本包；
3. 运行 15 个 `unittest` 用例；
4. 分别输出到 `artifacts/verify-a` 与 `artifacts/verify-b`；
5. 逐字节比较摘要、完整结论、SQLite 账本导出、原始 `.sqlite`、调度轨迹、编辑传播、快照替换和恢复账本。

也可手动运行：

```bash
python3.11 -m venv --without-pip .venv
curl -fsSL https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
.venv/bin/python -m pip install 'numpy==1.26.4'
.venv/bin/python -m pip install -e .
PYTHONPATH=src .venv/bin/tcp-observer all --out artifacts/demo
```

## 代码地图

- `src/tcp_observer/spec.py`：会话配置和对端脚本 JSON 强校验、JSON 行号定位、`E_SPEC`。
- `src/tcp_observer/scripting.py`：事件序号缺口 `E_SCRIPT`、重复 ACK 聚集 `E_DUPACK`。
- `src/tcp_observer/receipts.py`：o1 定宽行与 o2/o3 JSONL 编解码、默认补齐、来源/补齐字段/依据、`E_FORMAT`。
- `src/tcp_observer/transport.py`：`127.0.0.1` 回环套接字、真实连接/读取超时、端口占用 `E_PORT`、握手超时 `E_TIMEOUT`。
- `src/tcp_observer/peer.py`：以 `python -m tcp_observer.peer` 启动的真实子进程对端。
- `src/tcp_observer/fixedpoint.py`：定点除法、向上取整和 int64/窗口上限饱和规则。
- `src/tcp_observer/congestion.py`：慢启动、拥塞避免、三次重复 ACK、快速重传、超时、RTT/RTTVAR/RTO 递推和误差界。
- `src/tcp_observer/engine.py`：观测项、固定批次、封存窗口、批次区间引用、受影响子集和传播依据。
- `src/tcp_observer/scheduler.py`：全局在途批次上限、单连接配额、FIFO 排队、单调时钟和真实线程调度测试。
- `src/tcp_observer/ledger.py`：SQLite 表结构、单事务批次提交、误差界/调度轨迹/进度游标、规范化 JSONL 导出。
- `src/tcp_observer/recovery.py`：SIGKILL 后回滚恢复、从最后游标重建定点状态、重放和 `E_REPLAY` 定位。
- `src/tcp_observer/snapshot.py`：整数格式版本、脚本摘要、缩放口径、封存窗口、当前部分项、摘要识别、整体替换和只读备份。
- `src/tcp_observer/gui.py`：可选 Tkinter 编辑/校验/运行面板。
- `tests/test_observer.py`：15 个覆盖上述路径的本地确定性测试。

## 错误码

| 码 | 触发点 | 定位信息 |
| --- | --- | --- |
| `E_SPEC` | 字段类型/范围、MSS、scale、封存长度、dup count 与 `window_max` 不相容等 | `field`、JSON `line` |
| `E_FORMAT` | 未知回执版本、o1 宽度不足、o2/o3 缺字段、负 `duration_ms` | 首个非法回执 `line` 与字段 |
| `E_PORT` | 回环端口无法绑定/占用 | host、port、errno |
| `E_TIMEOUT` | 真实 TCP 握手 hello 或读取超过 `socket.settimeout` | timeout、event/port |
| `E_SCRIPT` | 脚本事件序号缺口 | `gap_start_seq`、`missing_count` |
| `E_DUPACK` | 窗口内重复 ACK 计数达到聚集阈值 | window、start/end、count、limit |
| `E_REPLAY` | 恢复重放与不间断推进不一致 | event_seq、item_id、字段路径 |

## 三版回执

o1 是 42 字符定宽 ASCII：

```text
O1 | start_seq:12 | end_seq:12 | ack_count:8 | cwnd:8
```

o2 是规范化 JSONL，增加：

```text
ssthresh, srtt_ms, peer_id, duration_ms
```

o3 再增加：

```text
scale, error_bound_scaled, affected[]
```

归一结果始终包含：

- `source_version`：`o1`、`o2` 或 `o3`；
- `filled_fields`：实际补齐字段；
- `fill_basis`：每个字段采用的显式默认规则；
- `source_line`：原始回执行号。

默认 scale 为 `1024`，默认 ssthresh 为 `16` 个 MSS，默认 SRTT 为 `24ms`；这些常量写在 `receipts.py`，可由 `NormalizationDefaults` 覆盖。

## 定点与误差规则

- 缩放因子为配置中的正整数 `K`，样本使用 `1024`。
- 所有非负除法采用 floor（向零/向下截断）；误差上界递推使用 ceil 保守取整。
- `cwnd`/`ssthresh` 饱和到 `window_max * K`；其他定点整数饱和到有符号 int64 上限。
- RTT：首次 `SRTT=R`、`RTTVAR=floor(R/2)`；之后按 `SRTT=(7SRTT+R)/8`、`RTTVAR=(3RTTVAR+|SRTT-R|)/4`。
- RTO 使用 `SRTT + 4*RTTVAR + 1ms` 时钟粒度，并向上取整为整数毫秒。
- 每个观测项输出本地单事件误差、五字段累积上界和可直接用于 o3 的 `error_bound_scaled`。

## 增量与封存

样本有 14 个事件、`batch_size=2`、`seal_length=4`，因此生成 7 个批次和 4 个窗口；前 3 个窗口封存，最后一个窗口保留部分观测。

编辑事件 `1012` 的延迟时：

```json
"affected_items": ["item-001012", "item-001013"],
"reused_item_count": 12,
"sealed_windows_byte_unchanged": true
```

每个受影响项都带有 `propagation_basis`，说明 RTT 更新、慢启动/拥塞避免规则、重复 ACK 或超时如何传播到下一项。新事件 `1013` 追加场景同样只生成一个新观测项，并复用前 13 项。

## 调度

样本调度约束为全局 2、单连接配额 1；`scheduler_demo.json` 额外用连接 A/B 展示全局 2 与每连接 1 的双重上限。逻辑单调时钟给出确定的 `wait_start_ms`、`wait_end_ms` 和 `wait_ms`。`SystemMonotonicClock` 使用真实 `time.monotonic_ns()`；线程测试证明真实并发任务不会越过全局或连接上限。

## 账本与恢复

每个批次在同一个 SQLite 事务中写入：

- `batches`：批次结论和摘要；
- `observation_items`：观测项结论；
- `observation_bounds`：逐观测项误差界；
- `schedule_trajectory`：调度轨迹；
- `progress`：连续提交游标；
- `recovery_events`：恢复时持久化的“未知未提交批次”和重放完成审计（不进入干净核心导出）。

所有核心记录带 `integer_format_version=1` 和 `content_digest`。`crash-demo` 在第 3 批插入后、提交前对自身发送 `SIGKILL`；下一次打开时 SQLite 自动回滚未提交事务。恢复报告显示最后完整提交批次为 2，未知并重跑批次为 3，重放 14 个观测项和 7 个批次全部匹配。

规范化账本导出按表、序号排序并使用 `sort_keys=True` 的 canonical JSONL，因此不依赖字典或哈希遍历顺序。

## 快照

`snapshot-current.json` 持久化：

- 整数快照/账本格式版本；
- `script_version`、`scale_version`；
- 脚本原文 SHA-256；
- 缩放口径和口径摘要；
- 已封存窗口集合；
- 当前开放窗口及部分观测项；
- 整体 `content_digest`。

v2 脚本把事件 `1001` 的延迟从 2ms 改为 3ms。重新打开时版本/摘要不一致，系统不做增量合并，而是整体替换当前快照，把旧文件复制为只读 `*.replaced-<digest>.bak.json`（权限 `0444`），并写只读 `*.stale.json` 标记旧封存窗口、误差界和观测视图陈旧。

## 实际环境说明

本机会话实际使用 `.venv/bin/python`：Python 3.11.0rc1、NumPy 1.26.4。Debian/Ubuntu 拆分了 `ensurepip`，因此当前环境采用 `python3.11 -m venv --without-pip .venv` 后通过 `get-pip.py` 引导；`scripts/reproduce.sh` 已包含该回退路径。

当前镜像的 Tk 运行库不完整：venv 中没有 `tkinter` 包，系统 Python shim 在导入时报告 `ImportError: libtk8.6.so: cannot open shared object file`。因此 Tk 面板代码已交付但未在本次无头环境启动；所有核心验收均通过 CLI、真实回环 TCP 和自动化测试完成。在正常 WSL Ubuntu 桌面组件完整的环境中，可运行 `PYTHONPATH=src .venv/bin/tcp-observer gui`。
