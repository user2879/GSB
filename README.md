# sharedbuf — 本地跨组件共享缓冲区访问平台

面向本地 Windows 环境（同样可在 Linux/macOS 运行）的纯 Python 实现，
仅依赖 NumPy，Tkinter 使用标准库自带组件，不使用 Docker 等容器。

## 结构

- `sharedbuf/protocol.py` — JSON 行协议与随会话变化的字段白名单（`wl1.<会话序号>`）。
  缺失/多余/类型不符字段均以 `E_FIELD` 拒绝，拒绝原因含字段路径与白名单版本。
- `sharedbuf/lease.py` — 租约状态机 `granted→active→expired/revoked`（恢复时另有
  终态 `unknown`）。每次迁移记录原因、序号进度与单调时刻；过期租约的缓存条目被
  真实回收（`reap()` 后 `cached()` 不再命中）；同一扇区存在未落定租约时 `grant`
  以 `E_BUSY` 拒绝，裁决层等待旧租约终态落定后才授予新租约。
- `sharedbuf/arbiter.py` — 并行裁决：全局并发上限（线程池）+ 每节点配额
  （信号量，配额耗尽排队不得绕过）+ 每扇区 FIFO 票锁按稳定序号串行化冲突申请；
  单次裁决有真实超时，超时重试不改变授予范围，各次尝试分列写入裁决轨迹。
- `sharedbuf/platform.py` — 平台核心：租约授予、按摘要重算期望并逐条比较
  （`E_VER` 含偏差节点、租约与首个差异环节）、恢复（`E_RECOVER`）与快照重放
  （`E_REPLAY` 定位首个不一致的租约/扇区/字段）。
- `sharedbuf/node.py` — 组件节点子进程：绑定唯一回环端口，NumPy 执行批量聚合，
  双向白名单校验。
- `sharedbuf/orchestrator.py` — 编排：端口占用 `E_PORT`、启动失败 `E_NODE`、
  握手超时 `E_HANDSHAKE`，均写入 JSON 结构化日志；启动仅改参数，不改样本与源码。
- `sharedbuf/store.py` — SQLite 持久化（租约、迁移、扇区版本、批量结果、裁决
  轨迹、快照），每次写入立即提交，强杀后仅依据完整落库记录恢复。
- `sharedbuf/ui.py` / `main.py` — Tkinter 界面或命令循环展示裁决轨迹。

## 运行

```bash
pip install -r requirements.txt
python main.py                 # CLI + 真实节点子进程
python main.py --mode ui       # Tkinter 界面
python main.py --no-nodes      # 本地执行器（不起子进程）
python main.py --recover --replay
```

## 关键语义

- 租约含过期毫秒数（单调时钟真实计时）、操作序号下界/上界与扇区集合；越界扇区
  `E_SCOPE`、越界序号 `E_SEQ`、非属主写 `E_OWNER`，拒绝无任何副作用。
- 批量计算为纯函数（`buffer.run_batch`），平台与节点由同一种子导出确定性数据，
  平台按摘要（租约标识、扇区集合、序号范围、扇区版本）重算并逐条比对。
- 恢复快照版本（白名单版本）与当前会话不一致时整体替换租约与扇区状态，不做
  增量合并；active 未落定租约恢复时标记 `unknown` 并要求重新申请，不生成退出报告。

## 验证结果（本机 Python 3.11 + NumPy 2.4）

`python3.11 -m pytest tests/ -q` → **18 passed**，覆盖：

- 租约越界/越序拒绝（`test_lease_scope_seq_and_expiry_recycling` 等）：E_SCOPE/E_SEQ，
  过期缓存真实回收，新租约在旧租约终态后授予。
- 字段白名单（`test_field_whitelist`、`test_node_rejects_malformed_message_over_wire`）：
  缺字段/多字段/错类型/白名单版本不符均 E_FIELD 且记录路径与版本，含真实 TCP 链路。
- 并行裁决预算（`test_budget_queueing_and_global_limit`、
  `test_adjudication_timeout_retry_same_scope`）：全局上限与节点配额可观测不超发，
  超时重试范围不变且分列记录。
- 强制终止恢复（`test_recovery_after_forced_kill`、`test_recover_detects_version_mismatch`、
  `test_snapshot_restore_wholesale_and_replay`、`test_replay_divergence_locates_first_mismatch`）：
  active 租约标记 unknown，扇区版本不一致报 E_RECOVER，快照版本不符整体替换，
  重放差异以 E_REPLAY 定位首个租约/扇区/字段。
- 编排故障（`test_port_occupied_reports_e_port`、`test_handshake_timeout_reports_e_handshake`、
  `test_node_exit_reports_e_node`）：E_PORT/E_HANDSHAKE/E_NODE 并写结构化日志。
- 端到端（`test_end_to_end_read_write_over_tcp`、`test_corrupt_node_flagged_e_ver`）：
  真实子进程经回环 TCP 完成读写并持久化，篡改节点被 E_VER 识别。

另用 `os._exit(0)` 模拟 kill -9 后经 `main.py --recover` 验证：悬挂租约
`Ls1-2` 标记为 `unknown`，重放 `ok`。
