# 数据库配置租约与并发控制诊断应用 — 技术说明

仅依赖 Python 3.11+ 标准库（`sqlite3`、`socket`、`tkinter`、`threading`、`json` 等），
不使用容器、不访问外部网络；协调层仅监听 `127.0.0.1` 高位端口，启动前探测端口空闲。

## 1. 模块结构

| 模块 | 职责 |
| --- | --- |
| `lease_diag/errors.py` | 结构化业务码 `E_POOL/E_ROLLBACK/E_LEASE/E_TIMEOUT/E_GAP/E_SNAPSHOT/E_REPLAY` 与带定位的异常 |
| `lease_diag/clock.py` | `RealClock`（`time.monotonic_ns` 真实单调时钟）与 `VirtualClock`（确定性重放/测试） |
| `lease_diag/model.py` | 池、请求、回滚边、租约、事件、裁决结果与显式状态迁移许可表 `TRANSITIONS` |
| `lease_diag/loader.py` | 池声明/NDJSON 请求流/回滚目标的强校验，错误整体拒绝、无部分加载 |
| `lease_diag/engine.py` | 租约状态机、真实计时到期、容量裁决、确定性抢占与回滚收敛 |
| `lease_diag/diagnostics.py` | 整数定点水位、单值 1 ulp 误差、比较 2 ulp 合成误差界与稳定决胜标注 |
| `lease_diag/ledger.py` | SQLite 单事务批次账本、水位/进度游标、OPEN 批次恢复语义、快照表 |
| `lease_diag/snapshot.py` | 版本识别、SHA-256 内容摘要、整体替换恢复（非增量合并） |
| `lease_diag/replay.py` | 按账本事件时间虚拟时钟重放，逐请求/副作用/水位/归属比对，`E_GAP/E_REPLAY` |
| `lease_diag/server.py` | 回环 TCP JSON 行协议、全局/单池信号量、真实 socket 超时与幂等重试 |
| `lease_diag/runtime.py` | 运行编排、批次持久化、强制终止后续跑（resume）、恢复与重放 |
| `lease_diag/samples.py` | 确定性样本生成与三类异常注入 |
| `lease_diag/ui.py` | Tkinter 仪表盘；无 Tk 环境自动降级为稳定排序的文本仪表盘 |
| `lease_diag/cli.py` | 命令行入口 `generate / run / recover / demo` |

## 2. 业务要求到实现的映射

1. **强校验整体拒绝（E_POOL/E_ROLLBACK）**：`loader.py` 收集全部问题后一次性抛出
   `LeaseError`，定位精确到 `文件#pools[i]`、`文件#line[i]`、`文件#rollback[i]`。
   回滚链按键做有环检测并打印环路径；目标版本必须存在于请求流。
2. **状态机与水位账本**：许可表 `TRANSITIONS` 仅允许
   `granted/renewed -> {renewed, expired, revoked, rolledback}`，终态无出边。
   续约必须是当前持有者、在到期前、续约计数递增；到期由 `Engine.tick(monotonic_now)`
   基于真实单调时钟判定；到期/撤销/回滚立即释放容量并在批次事务中入账。
   同一 `(池, 键)` 任意时刻至多一个活跃租约（`active_index` 保证）。
3. **确定性抢占与回滚收敛**：容量耗尽且存在严格更低优先级候选时，按
   `(优先级, 最近续约边界, 租约 id)` 稳定选择受害者；撤销事件落在最近续约边界并保留
   全部已提交续约。存在回滚边时，该键所有未过期租约整体收敛到目标修订并置
   `rolledback`；无可抢占者或竞争落败返回 `E_LEASE` 并携带冲突租约 id。
4. **三重约束并行调度**：全局信号量限制运行授予总数、每池信号量限制单池运行数
   （池配额 `quota=min(quota,capacity)`），客户端 `socket.settimeout` 产生真实超时，
   等待开始/结束单调毫秒写入 QUEUE 事件与队列统计；超时按 `--max-retries` 重试，
   请求 id 不变故裁决幂等，单次授予语义不变。
5. **整数定点诊断与误差界**：默认缩放因子 10000，比例 `n*scale//d` 显式向零截断；
   单值绝对量化误差 1 ulp（`1/scale`），两值比较合成误差界 2 ulp（`2/scale`）。
   水位差落入界内时按 `(ratio 降序, capacity 降序, pool_id)` 决胜并标注
   `error_influenced=true`。
6. **账本持久化与强制终止恢复**：每批次在单个 SQLite 事务内写事件、租约状态、
   水位与进度游标；批次先插 `OPEN` 再在同事务置 `COMMITTED`。强制终止后 OPEN 批次
   无任何 durable 效果，恢复报告将其列为 `unknown_open_batches` 并要求重跑，
   不生成退出报告；`--resume` 通过 `reset_to_last_committed` 删除未知批次并从游标续跑。
7. **快照版本识别/整体替换/重放校验**：快照携带池集合版本、回滚版本、缩放口径版本、
   水位、租约集合、账本游标及 SHA-256 摘要。任一不符抛 `E_SNAPSHOT` 并从账本整体
   重建（`whole_replace_leases`）。重放按账本事件时间驱动虚拟时钟，比对请求裁决、
   副作用事件序列（EXPIRE/REVOKE/ROLLBACK）、批次水位与租约归属，首个不一致定位到
   请求 id 与字段并以 `E_REPLAY` 报明。
8. **Tkinter 界面/异常注入/可重复演示**：仪表盘只读账本，固定排序、刷新不写库；
   `--inject rollback_cycle|missing_renewal|snapshot_digest` 分别产生
   `E_ROLLBACK/E_GAP/E_SNAPSHOT`，并写入 `structured.log`（JSON 行）。

## 3. 业务码

| 码 | 触发场景 |
| --- | --- |
| `E_POOL` | 池声明或请求流字段/单调性/引用校验失败 |
| `E_ROLLBACK` | 回滚目标不存在或回滚链成环 |
| `E_LEASE` | 持有权冲突、容量不足且无可抢占者、竞争落败（附冲突租约 id） |
| `E_TIMEOUT` | 真实 socket 授予超时（可重试，重试幂等） |
| `E_GAP` | 账本引用的请求（如续约）在当前请求流缺失 |
| `E_SNAPSHOT` | 快照版本或内容摘要与当前会话不一致 |
| `E_REPLAY` | 确定性重放与账本在裁决/副作用/水位/归属上出现首个不一致 |

## 4. 运行方式（Windows）

```bat
python -m lease_diag --workdir workspace\demo demo
python -m lease_diag --workdir workspace\demo --inject rollback_cycle demo
python -m lease_diag --workdir workspace\demo --inject missing_renewal demo
python -m lease_diag --workdir workspace\demo --inject snapshot_digest demo

python -m lease_diag --workdir workspace\kd generate
python -m lease_diag --workdir workspace\kd --hard-exit-batch 2 run
python -m lease_diag --workdir workspace\kd --resume run
python -m lease_diag --workdir workspace\kd recover
```

`scripts/*.bat` 封装了上述可重复演示。沙箱无 Tk 运行库时使用文本仪表盘；
Windows 官方 Python 自带 tkinter，可在 `demo --ui` 下打开图形界面。

## 5. 实际验证结果

- 单元/集成测试：`python -m unittest discover -s tests`，**38 个测试全部通过**。
- 确定性样本演示：6 个批次全部 COMMITTED，恢复重放 18 个请求 `all_match=true`，
  活跃租约 5、总租约 7；`db.conn` 经抢占回滚收敛到目标修订，`cache.ttl` 抢占后重新授予。
- 水位误差：等比例池（P_A/P_B/P_C 均 10000 fp）按稳定决胜排序，P_B/P_C 标注
  `error_influenced=true`，合成误差界 2 ulp。
- 强制终止：批次 2 中途 `os._exit(87)` 后账本仅批次 1 COMMITTED、批次 2 OPEN、
  游标停在 R03；`--resume` 后 6 批次全部 COMMITTED 且重放完全一致。
- 异常注入：成环→`E_ROLLBACK`（含环路径）；缺续约 R09→`E_GAP`；
  快照缩放版本+摘要篡改→`E_SNAPSHOT` 后整体替换并重放一致；均写入结构化日志。
