# 声明式部署收敛诊断组件 — 技术说明

## 组件结构

| 模块 | 职责 |
| --- | --- |
| `convd/scene.py` | 场景与执行器集合 JSON 加载、强校验（`E_SCENE`，稳定报告首个非法字段）、规范摘要 |
| `convd/wire.py` | 回环 UDP 数据报编解码（脉冲/回执/错误/探活），长度自洽校验，`E_WIRE` 报告出错偏移 |
| `convd/executor.py` | 执行器子进程：确定性版本推进、重复脉冲幂等、轮次倒退回 `E_STALE` |
| `convd/orchestrator.py` | 脉冲编排：时钟节奏、租约判定、并发上限与配额、真实套接字超时重试、时序异常识别 |
| `convd/ledger.py` | SQLite 账本：单事务逐轮提交、进度游标、内容摘要、会话快照版本识别整体替换 |
| `convd/replay.py` | 强制终止恢复（未提交轮次标记未知并重跑）与重放校验（`E_REPLAY` 定位首个分歧） |
| `convd/aggregate.py` | NumPy 收敛矩阵聚合校验，与账本摘要逐执行器对账 |
| `convd/cli.py` | `run` / `resume` / `verify` / `summary` 命令行 |

## 确定性设计

- 执行器版本规则为声明式单调规则：`version = round`；状态摘要
  `sha256(scenario_id|executor_id|v<version>|target_digest)`，推进时长
  `advance_ms(seed, round)` 为整数种子的纯函数。
- 执行器收到重复轮次脉冲时回放缓存回执（逐字节一致、不重复推进）；轮次倒退回 `E_STALE`。
- 账本内容摘要只覆盖逻辑字段（版本号、状态摘要、收敛结论、异常集合、推进时长），
  真实时钟（发送/到达/排队等待的单调毫秒）持久化在 `timing` 列但不计入摘要，
  因此相同输入在任意执行器启动顺序、任意主机时序下得到逐字节一致的轨迹与异常集合
  （`tests/test_orchestrator.py::TestDeterminism` 验证两次运行摘要相同）。
- 租约判定为真实计算：`receipt.obs_mono_ms <= 本轮首次发送单调毫秒 + lease_ms`；
  执行器在推进前真实睡眠 `advance_ms`，租约窗口物理流逝（`test_lease_expiry_detected_with_real_timing`）。

## 线路协议（大端，长度前缀 UTF-8）

- 脉冲 `CP1P`：场景标识、轮次 u32、租约 u32、发送单调毫秒 i64、目标状态摘要。
- 回执 `CP1R`：执行器标识、轮次 u32、观测序号 u32、观测时刻 i64、状态摘要。
- 错误 `CP1E`（含 `E_STALE`）、探活 `CP1N`/`CP1O`。
- 解码校验长度自洽、字段越界与截断，均以 `E_WIRE` 报告字节偏移；轮次/场景不匹配、
  观测序号倒退在编排器侧以 `E_WIRE` 拒绝。

## 调度与重试隔离

- 全局在途上限（信号量）+ 每执行器在途配额（每执行器信号量）；配额等待记录起止单调毫秒。
- 单次脉冲真实 `socket` 超时（`receipt_timeout_ms`）；超时重试不改变本轮租约锚点
  （租约始终锚定本轮首次发送时刻），重试/超时事件写入独立 `retries` 表，与收敛结论分离。
- 重复观测序号：保留首次回执，被覆盖回执登记为 `duplicate_receipt`；
  已结束轮次的迟到回执登记为 `late_receipt`，不修改已提交结论。

## 时序异常

连续轮次间识别：`version_regression`（版本回退）、`digest_jump`（状态摘要跳变，
对照声明链重算）、`obs_seq_gap`（回执序号缺口）；叠加 `late_receipt`、
`duplicate_receipt`、`wire_error`。异常集合按 `(executor_id, kind)` 稳定排序后入账。

## 账本与快照

- 每轮轨迹、结论、异常、重试与进度游标在同一 SQLite 事务提交；账本携带整数格式版本
  （`LEDGER_FORMAT_VERSION=1`）与内容摘要（随游标同事务更新）。
- 快照（`<db>.snapshot.json`）持久化场景版本、预算参数版本、执行器集合版本、逐轮结论
  与账本摘要；重新打开时任一字段不一致即整体替换，旧快照保留为只读
  `<db>.snapshot.backup.json`（chmod 444），差异项按字段名稳定排序列出。

## 恢复与重放

- 强制终止后仅以已提交游标恢复；游标之后轮次标记为未知并要求重跑，不生成终止时退出报告。
- `verify` 按账本记录的场景/预算版本重新编排已提交轮次，逐轮逐执行器比对收敛判定、
  异常集合与轨迹字段，首个分歧以 `E_REPLAY` 报明轮次、执行器标识、字段与双方取值。

## 验证结果（本机实测）

- `python3 -m unittest discover -s tests`：41 项测试全部通过
  （场景校验 9、线路协议 9、执行器 6、编排/租约/重试/确定性 8、账本与快照 5、
  重复/迟到回执 2、强制终止恢复与重放 2）。
- `python3 demo.py`：
  1. 6 轮 × 3 执行器全部在租约内收敛；`exe-beta` 第 3 轮首次脉冲确定性丢弃 →
     真实超时 + 重试分离记账，本轮仍收敛且不影响其他轮次；两次运行账本摘要逐字节一致
     （`0b2fb4cc…d2ed68`）。
  2. 租约收紧至 2ms、执行器推进 30ms：全部轮次真实判定 `lease_expired`。
  3. 10 轮运行在游标 4 处 `SIGKILL` → 恢复标记 5–10 轮未知并重跑 →
     重放 10 轮全部一致（`digest_match: true`），NumPy 聚合校验通过。
- 篡改已提交结论后 `verify` 抛出 `E_REPLAY` 并定位首个不一致的轮次/执行器/字段
  （`tests/test_recovery.py::test_tampered_ledger_fails_replay_with_first_mismatch`）。

## 运行方式

```bash
python3 -m convd.cli run    --scenario samples/scenario.json --executors samples/executors.json --db out.db
python3 -m convd.cli resume --scenario samples/scenario.json --executors samples/executors.json --db out.db
python3 -m convd.cli verify --scenario samples/scenario.json --executors samples/executors.json --db out.db
python3 -m convd.cli summary --db out.db
```

仅依赖 Python 3.10+ 标准库（`sqlite3`/`socket`/`subprocess`/`threading`）与 NumPy；
全部通信限于 127.0.0.1 回环 UDP，无容器、无外部网络。
