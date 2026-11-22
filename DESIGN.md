# GPU 配额与分片抢占调度诊断工作台 — 技术说明

面向 Windows 本地环境、零外部账号/网络依赖的确定性调度诊断应用。运行期仅使用
Python 标准库（`json`、`sqlite3`、`tkinter`、`concurrent.futures`）加 NumPy，
pytest 用于测试。Tkinter 为惰性导入，无 Tk 运行时（如本 WSL 沙箱）自动降级为
等价的无头文本快视图，Windows 官方安装包自带 Tk 可直接开窗。

## 1. 目录与模块

- `gpu_diag/errors.py`：业务码（`E_DECL/E_MEMORY/E_QUOTA/E_CYCLE/E_LEDGER/
  E_REPLAY/E_PREVIEW/E_STATE`）与追加式 JSON-lines 结构化日志（按键排序）。
- `gpu_diag/models.py`：`Pool/Task/Shard/SessionParams/Declaration` 数据模型。
- `gpu_diag/loader.py`：JSON 加载与“先收集全部违规、再整体拒绝”的强校验。
- `gpu_diag/fixedpoint.py`：NumPy 整数定点水位比例（floor 截断，无浮点）。
- `gpu_diag/digest.py`：唯一的规范化 JSON 字节序与 SHA-256 哈希/哈希链。
- `gpu_diag/sim/worker.py`：真实本地线程池（`ThreadPoolExecutor`）分片执行。
- `gpu_diag/sim/propagation.py`：分片图/任务图失效传播，稳定拓扑序、与遍历顺序无关。
- `gpu_diag/sim/engine.py`：tick 驱动的确定性调度引擎（准入/排队/抢占/恢复/水位）。
- `gpu_diag/persist/ledger.py`：SQLite 单事务原子账本 + 派发意图表 + 未知分片。
- `gpu_diag/persist/snapshot.py`：版本化快照、整体替换差异、`E_REPLAY` 重放定位。
- `gpu_diag/session.py`：运行、强制终止恢复、权威重算、快照恢复编排。
- `gpu_diag/cli.py`：命令行入口与四类异常注入。
- `gpu_diag/ui/app.py`：账本缓存快速视图（Tk 窗口 / 无头文本）。
- `samples/`：确定性样本；`tests/`：36 个 pytest 用例。

## 2. 调度模型与三重约束

虚拟单调时钟以固定 tick（10ms）推进；分片在途跨多个 tick，到达时刻可落在轮间，
从而真实表达“在途占用 + 分片边界抢占 + 逐轮水位”。每个 tick（一轮）：
1. 完成到时分片（结果由工作线程 future 取回）；
2. 仅在有分片完成释放资源的窗口恢复被抢占任务（防止同优先级抖动）；
3. 按 `(-优先级, 提交时刻, 任务id, 分片序号)` 排序就绪集合并逐一准入；
4. 准入同时校验三重约束：全局 GPU 配额、每池并发任务上限、每任务显存预算
   （外加每池 GPU/显存容量）；不足则排队并记录等待起止单调时钟；
5. 高优先级任务无法启动且存在更低优先级在途任务时，按显式全序选择受害者，
   在最近分片边界安全挂起，释放配额当轮即可被高优先级分片占用；
6. 汇总逐轮水位并追加哈希链。

抢占受害者选择全序：优先级升序 → 最近启动时刻（start_ms 大的优先）→
GPU 占用降序 → 任务 id 字典序。优先级严格不降序，已完成分片状态与摘要保持不变。

被抢占任务恢复时只重跑 `next_index` 之后的未完成分片，复用已完成分片摘要，
已完成分片不再产生任何 GPU 配额占用。

## 3. 定点水位口径

`ratio_q = floor(used * scale_factor / capacity)`，`scale_factor` 为 10 的整数幂，
全程 `numpy.int64` 整数运算，截断方向显式记录为 `floor`。量化误差上界：
`0 <= 精确值 - 量化值 < 1`（缩放整数单位），即比例单位绝对误差 `< 1/scale_factor`。
每轮输出占用、余量、池内任务数、显存占用/余量与定点比例等全部整数中间项。

## 4. 确定性与逐字节一致

- 唯一规范化字节序 `canonical_bytes`（`sort_keys`、固定分隔符、UTF-8）同时用于
  SQLite 账本字段、快照文件与哈希链；tuple 在引擎出口统一转为 list。
- 决策负载只记录确定性摘要与“是否离开主线程”的布尔证据，不记录线程名等易变量。
- 时序来自声明式 `duration_ms` 驱动的虚拟时钟，线程池只承担真实计算投射。
- 同一声明与提交时刻重复运行，快照文件与轮次哈希链逐字节一致（测试已断言）。

## 5. 持久化、强制终止与恢复

每轮在**同一个 SQLite 事务**写入轮次记录、分片轨迹、抢占记录与进度游标。
`dispatch_journal` 在轮次提交前持久化派发意图：

- 正常提交：意图行标记 committed 并清除；
- `kill -9`（由 `--crash-after-round` 模拟）：意图残留而轮次永不提交，
  重启时 `reconcile_crash` 将这些分片标记为 `UNKNOWN` 并要求重跑；
- 绝不按挂起位置补算配额，也不声称能生成强制终止时刻的退出报告。

## 6. 快照与重放

快照携带池/任务/缩放口径版本、各轮水位、抢占序列、分片结局摘要、账本游标与
内容摘要。重开时版本或摘要不一致走**整体替换**并稳定列出差异项（非增量合并）。
重放按快照记录的任务版本与缩放因子重算剩余分片，逐项比对水位、抢占序列与结局，
首个偏差定位到轮次/分片/字段并以 `E_REPLAY` 报明；界面侧权威重算偏差用
`E_PREVIEW`。账本哈希链损坏用 `E_LEDGER`。

## 7. 异常注入（只改输入文件/参数）

`python -m gpu_diag.cli inject {memory,quota,cycle,ledger}`

- `memory`：任务每卡显存改为超过池容量 → `E_MEMORY`；
- `quota`：池配额份额之和超过 `quota_share_cap` → `E_QUOTA`；
- `cycle`：任务依赖成环 → `E_CYCLE`；
- `ledger`：篡改已提交轮次水位破坏摘要链 → `E_LEDGER`。
四类事件均以结构化 JSON 行写入 `logs/events.jsonl`，业务退出码为 2。

## 8. 常用命令

```bash
python -m gpu_diag.cli init-samples
python -m gpu_diag.cli run                      # 完整确定性会话
python -m gpu_diag.cli run --crash-after-round 8
python -m gpu_diag.cli run --fail-shard T1#1
python -m gpu_diag.cli recover                  # 崩溃恢复/未知分片
python -m gpu_diag.cli recompute                # 权威重算（E_PREVIEW）
python -m gpu_diag.cli restore                  # 快照整体恢复+E_REPLAY
python -m gpu_diag.cli verify-ledger            # 哈希链校验（E_LEDGER）
python -m gpu_diag.cli ui                       # Tk 窗口（无 Tk 则无头视图）
python -m pytest -q                             # 36 个测试
```

Windows 一键演示：`run_demo.bat`；类 Unix：`run_demo.sh`。

## 9. 实际验证结果（本机 WSL / Python 3.11.0rc1）

- `pytest -q`：36 passed，覆盖强校验、三重约束、定点误差界、真实线程投射、
  分片边界抢占确定性、失效传播顺序无关、崩溃未知分片、快照逐字节一致、
  `E_REPLAY/E_PREVIEW/E_LEDGER` 与四类注入业务码。
- 默认样本 59 个 tick，抢占序列稳定（T2/T5 在分片边界挂起并恢复复用）。
- 专用抢占样本（`write_preemption_samples`）中 HIGH 在提交后 1 个 tick 内迫使
  LOW、MID 依次挂起，完成后二者只重跑未完成分片。
- 连续两次 `run` 产出的 `snapshot.json` `cmp` 完全相同；`recompute` 返回
  `byte_identical=true`。
