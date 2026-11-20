# 数据库租户资源配额仲裁工作台

面向 Windows / Linux 本地环境的**领域可视化与诊断工作台**。用本地 JSON 声明租户与资源池、
NDJSON 描述事务请求流，对每条请求作出 **授予 / 排队 / 拒绝** 三态仲裁，并把结局、配额扣减、
排队轨迹持久化到 SQLite，支持三条执行路径、自适应选型、强制终止恢复与快照重放校验。

仅依赖 Python 标准库（`sqlite3`、`tkinter`、`hashlib` 等）与 NumPy（确定性数组校验），
不使用 Docker 或任何容器。

## 1. 环境准备

- Python 3.11+；确认 `python -c "import sqlite3, tkinter"` 可用。
- NumPy ≥ 1.26：`pip install "numpy>=1.26"`。
- Windows 请使用 python.org 官方安装包（默认勾选 tcl/tk，自带 tkinter）。
- 最小化 Linux 若缺 tkinter，需 `sudo apt-get install python3-tk`；**核心仲裁与 CLI 不依赖 tkinter**，
  缺失时 GUI 给出明确提示，命令行功能完全可用。

本仓库实测环境：Python 3.11.0、NumPy 1.26.4、sqlite 3.37.2（当前沙箱无 root，未装 tkinter，
故 GUI 采用延迟导入并已验证降级提示路径）。

## 2. 目录结构

```
quota_arbiter/
  errors.py       业务异常与稳定业务码 E_SPEC/E_ORDER/E_PRIORITY/E_LEDGER/E_REPLAY
  models.py       Tenant / Pool / Request / Outcome / QueueEvent 数据模型
  util.py         确定性摘要(SHA-256)、保持时长、JSONL 结构化日志
  loader.py       JSON/NDJSON 加载与强校验（首个出错对象 + 字段）
  engine.py       确定性仲裁状态机：授予/排队/拒绝、释放、稳定唤醒
  paths.py        三条执行路径的批次计划（共享同一引擎与规范序）
  selector.py     按规模自适应的默认路径选择、工作量估算、阈值决胜
  consistency.py  NumPy 结局序列/终态/排队轨迹逐字节校验
  ledger.py       SQLite 账本：批次原子事务、摘要链、崩溃恢复、篡改检测
  snapshot.py     会话快照：版本识别、整体替换、只读备份、重放校验
  generator.py    确定性样本生成与三类异常注入（只改数据/参数，不改源码）
  runner.py       运行编排：落账、恢复重跑、快照重放、三路径一致性
  cli.py          命令行入口
  gui.py          tkinter 可视化诊断台（延迟导入）
tests/            pytest 测试套件（26 个用例）
run_arbiter.py    CLI 便捷入口；run_arbiter.bat 为 Windows 包装
run_gui.py        GUI 入口；demo.py 端到端验收脚本
```

## 3. 数据模型与显式判决规则

### 声明
- 租户 `tid / quota(正整数配额份额) / concurrency_cap(正整数并发上限) / burst(非负整数突发额度)`
- 资源池 `pid / capacity(正整数总容量) / priority_order(按稀缺度排序的租户优先级序)`
- 请求 `rid / tid / pid / demand(正整数需求) / arrival(非负毫秒到达) / priority(非负，越小越优先)`

### 到达请求的显式判决顺序（见 `engine.py::_admit`）
1. **租户累计配额不足** → 拒绝 `TENANT_QUOTA_EXHAUSTED`。
2. **在途并发达到上限** → 排队，立即预占租户配额，记录入队时刻。
3. 池常规容量足够 → 扣租户配额并**常规授予**（占池容量）。
4. 池常规容量不足时分类：
   - 突发额度为 0 → 拒绝 `POOL_CAPACITY_EXHAUSTED`；
   - 有突发但不足以覆盖需求（或需求超过池容量）→ 拒绝 `BURST_INSUFFICIENT`；
   - 突发足够 → 扣租户配额并**突发授予**（扣突发额度，不占池常规容量）。

配额与突发是**累计预算**（授予即扣，不回填）；池容量与并发是**运行期占用**，请求完成时释放。
样本没有显式完成时刻，保持时长由请求字段经确定性函数 `hold_duration_ms` 推导，
因此完成时刻与释放顺序在任何路径下都一致。

### 容量释放后的稳定唤醒（见 `engine.py::_wake_pool`）
唤醒候选按显式稳定键排序，**不依赖 list/dict/set 的遍历顺序**：

```
(资源池内租户优先级序 rank, 请求优先级 priority, 到达时刻 arrival, 请求标识 rid)
```

每次释放取队首尝试授予，不能授予则记录 `STAY` 并停止本轮；唤醒输出被唤醒请求的
`rid / 等待毫秒 / 判决依据`。测试 `test_wake_independent_of_internal_queue_order`
证明队列内部存储顺序被打乱时出队序列仍然相同。

## 4. 三条执行路径为何逐字节一致

确定性的根本来源：**三条路径都以同一全局规范序 `(arrival, rid)` 驱动同一个状态机**，
引擎内部的完成/唤醒事件完全由到达时刻决定。路径之间只改变**批次边界与统计口径**：

| 路径 | 批次定义 | 提交粒度 |
| --- | --- | --- |
| `realtime` 逐请求实时 | 每条请求一个批次 | 每请求一事务 |
| `batch` 按到达时刻分批 | 同一 `arrival` 的请求一批 | 每时间桶一事务 |
| `merged` 按租户归并批量 | 按排序租户序聚合，提交边界落在各租户组在规范流中的最后位置，批次为规范流连续切片 | 每租户一事务 |

归并路径额外输出每批 `request_count / skipped_count（被跳过的其他租户请求数）/ merge_order`，
`merge_order` 取排序后的租户标识序列，**绝不依赖字典或哈希遍历顺序**。
三路径结局数、配额终态、排队轨迹通过 NumPy 数组逐元素比较，并对比较结果计算 SHA-256，
`consistency` 命令输出三份摘要，完全一致时 `identical=true`。

## 5. 自适应默认路径选择（`selector.py`）

仅依据租户数 T、请求数 N、资源池数 P 与显式常量估算工作量：

```
realtime ≈ 3.0·N·(1+0.08P)
batch    ≈ (12P + 1.4N)·(1+0.08P)
merged   ≈ (18T + 1.1N)·(1+0.08P)
```

按工作量升序选择；当最优与次优相对差 `< 10%`（`THRESHOLD_RATIO`）时触发**稳定决胜**
（固定路径名序 `realtime < batch < merged`），并在理由中标注 `threshold_influenced=true`。
相同输入永远得到相同选择，且该结论与三路径一致性校验并存（选型不影响结果，只影响批次口径）。

## 6. SQLite 账本、恢复与快照

### 账本（`ledger.py`）
- 表：`meta / tenant_state / pool_state / outcomes / queue_events / batches / unknown_requests`。
- 每个完整批次在**同一事务**写入逐请求结局、租户配额/突发/在途、池容量、排队轨迹、进度游标。
- 一条请求全生命周期只保留一条 `outcomes` 记录（入队落 QUEUE，唤醒后同 rid 更新为 GRANT）。
- 每批计算内容摘要（含结局与轨迹的完整规范元组）并形成链式 `head_digest`，
  `meta` 携带整数 `format_version` 与 `spec_digest`。
- 校验时从**实际表内容重算**每批摘要再比对，任何对结局/状态/轨迹的篡改都会以
  `E_LEDGER` 报出并定位到具体请求标识。

### 强制终止恢复
`crash` 命令在指定批次的事务提交前中断（事务回滚，连接关闭）。重启后：
- 只承认已完整提交的批次，恢复到最晚提交的请求标识；
- 仲裁中未提交的请求写入 `unknown_requests` 并要求重跑（全新账本完整重算）；
- 系统**不生成强制终止时刻的退出报告**，仅输出恢复点与未完成请求数。

### 会话快照（`snapshot.py`）
快照持久化租户集合版本、资源池版本、默认路径选择、各租户配额/突发/在途终态、
池容量终态、已处理游标、完整结局与排队轨迹，并带整数格式版本与内容摘要。
重新打开时先比对版本与摘要：不一致则**整体替换**（非增量合并），旧快照复制为
`*.readonly.bak` 只读备份；随后按快照记录的路径重算并逐项比较，
首个不一致以 `E_REPLAY` 报明请求标识与字段。

## 7. 异常注入与业务码

注入只生成被污染的数据文件或篡改账本参数，**不修改源码**：

| 注入 | 业务码 | 触发方式 | 追溯 |
| --- | --- | --- | --- |
| 到达时刻倒退 | `E_ORDER` | 改写 NDJSON 第二条 arrival 早于第一条 | 请求 rid + `arrival` |
| 优先级序缺失租户 | `E_PRIORITY` | 改写 spec 优先级序为不存在租户 | 池 pid + `priority_order` + missing_tid |
| 账本摘要不符 | `E_LEDGER` | 改写一条结局但不更新摘要链 | 首个不一致 rid + 摘要字段 |
| 声明/请求强校验 | `E_SPEC` | 任意非法输入整体拒绝 | 首个出错对象标识 + 字段 |
| 重放不可复现 | `E_REPLAY` | 快照损坏/版本不符/重算分歧 | 请求 rid + 字段 |

所有异常与批次事件写入 JSONL 结构化日志（默认 `<workdir>/events.jsonl`）。

## 8. 快速开始（Windows）

```bat
python run_arbiter.py generate --workdir data --tenants 8 --pools 3 --requests 500
python run_arbiter.py consistency --workdir data
python run_arbiter.py run --workdir data
python run_arbiter.py crash --workdir data --crash-after-batch 2
python run_arbiter.py restore --workdir data
python run_arbiter.py inject --workdir data order
python run_arbiter.py inject --workdir data priority
python run_arbiter.py inject --workdir data ledger
python run_gui.py
```

也可一键端到端验收：`python demo.py`（产物在 `samples/demo/`，报告
`samples/demo/acceptance_report.json`，日志 `logs/demo-events.jsonl`）。

## 9. 实测验证结果

- `python -m pytest -q`：**26 passed**（强校验、三态判决、稳定唤醒、三路径一致性、
  选择器确定性、账本摘要链、强制终止恢复、快照整体替换/只读备份/E_REPLAY、三类注入）。
- `python demo.py`：三路径 `identical=true`，默认路径选择含阈值决胜标注，
  快照重放通过，强制终止恢复正确标注未知请求，三类异常码分别为
  `E_ORDER / E_PRIORITY / E_LEDGER`。
- 大规模样本（300 租户 / 16 资源池 / 30000 请求）：三路径结局、终态、轨迹摘要完全一致，
  端到端一致性校验耗时约 7 秒；批次数 realtime=30000、batch=506、merged=300。

## 10. 设计取舍说明

- **单引擎 + 多批次口径**：若让归并路径真正按租户分组顺序驱动引擎，会破坏基于到达时刻的
  事件仿真，导致结果分歧。本实现让引擎始终按规范序推进，归并只决定提交边界，
  从根上保证三路径逐字节一致，同时保留“先聚合、逐租户成批、按序合并”的批次与统计语义。
- **完成事件统一在流末排空**：完成/唤醒只取决于到达时刻，与批次边界无关，
  避免不同提交粒度造成事件提前刷新。
- **结局最终态统一 UPSERT**：排队请求在唤醒后由 QUEUE 变为 GRANT，按 rid 更新同一条记录，
  杜绝一条请求两条结局。
