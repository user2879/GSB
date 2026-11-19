# RTS 资源征收与单位调度回放工作台 — 技术说明

## 1. 目标与范围

本地、确定性、无外部服务依赖的即时战略对局回放工作台。固定种子程序化产出事件流与规则
配置，逐 tick 合成权威资源/单位状态，在 PySide6 交互时间轴上回放，并输出可独立复算的
账本与哈希报告。在线帧驱动路径与离线全量扫描路径共用同一合成引擎，保证逐字段一致。

仅依赖 `numpy`（随机数与生成器）与 `PySide6`（界面），不使用容器、不访问网络服务。

## 2. 目录结构

```
rts_replay/
  errors.py       业务错误码、硬错误 RtsError、软错误 TickError
  hashing.py      规范化 JSON（排序键、紧凑分隔）与 SHA-256
  rng.py          (chain_version, match_seed, tick, rng_seed_delta) 子流派生
  events.py       事件模型、JSONL 严格校验、确定性提交序键 (tick, actor, seq)
  rules.py        阵营规则与 E_RULE_RANGE 校验
  state.py        GameState/FactionState/QueueItem、状态哈希、快照版本
  generator.py    固定种子样本集（5 个场景）与清单/内容哈希
  engine.py       逐 tick 合成引擎、账本、每 tick 状态哈希、软错误收集
  store.py        原子写快照、最近快照恢复、E_SNAPSHOT_STALE
  player.py       帧驱动在线回放、预算降级、跳转重建、资源曲线历史
  verifier.py     在线/离线核对、RNG 漂移检测、跳转等价、错误审计
  offline.py      离线全量合成与报告落盘
  demo.py/demo_clock.py  固定序列演示与确定性脚本时钟
  gui/main_window.py     PySide6 时间轴窗口
  cli.py/__main__.py     命令行入口
tests/            unittest 测试套件（34 个用例）
workspace/        运行期可写目录：samples/ snapshots/ gui_snapshots/ reports/
```

## 3. 确定性设计

### 3.1 规范化序列化与字节一致

所有落盘数据经 `hashing.canonical_dumps` 输出：键排序、紧凑分隔符、UTF-8、禁止 NaN。
状态哈希为规范化状态字典的 SHA-256。两次相同种子生成、相同预算演示，全部 89 个产物
文件逐字节相同（见第 9 节验证）。文件写入采用“临时文件 + fsync + 原子 rename”，外部
强杀进程不会留下半文件。

### 3.2 事件与提交序

每行事件含 `tick、actor、action、target、amount、rng_seed_delta、seq`。`seq` 为 actor
本地单调序号，使序号断裂在分组重排后仍可检测。引擎先按 tick 分组，tick 内按
`(actor, seq)` 升序提交，结果与输入顺序、线程调度无关。

### 3.3 随机数派生链

每个事件独立子流：

```
seed = SHA256((RNG_CHAIN_VERSION, match_seed, tick, rng_seed_delta))[:8] 转整数
rng  = numpy.random.default_rng(seed)
```

子流不依赖此前事件消耗了多少随机数，因此快照跳转、部分重放都能复现同一随机值。账本为
每个事件记录 `rng_value`。将链版本 +1 后重新合成，所有消耗随机性的 tick 哈希改变，首个
受影响 tick 以 `E_RNG_DRIFT` 报告（核对器自动执行该对照）。

### 3.4 无前瞻

引擎只读取“当前 tick 分组”内的事件，`advance_to(n)` 不接触 tick>n 的数据。测试用
“截断到 tick 20 的前缀事件流”与“完整流推进到 20”状态哈希相等来锁定该性质。

## 4. 合成语义

- `collect`：随机扰动 `rng_value∈[-5,5]`，实收 `max(0, amount+rng_value)`；超出阵营
  `resource_cap` 时截断并记 `E_RESOURCE_OVERFLOW`（软错误，账本仍留痕）。
- `produce`：成本 `unit_cost*amount`。并发命令达到 `max_concurrent_commands` 或资源不足
  时记 `E_QUEUE_BLOCKED` 且不入队；成功则扣费、入队，`ready_tick=tick+queue_ticks`，
  质量标签来自独立 RNG。
- `move`：无空闲单位记 `E_QUEUE_BLOCKED`；否则按 `min(amount, idle)` 调度，RNG 命中 0
  时全损，否则转部署单位。
- 每 tick 末推进队列：到期订单转为空闲单位并释放并发名额。

软错误不终止回放，硬错误（畸形输入、规则越界、快照过期、非法预算）抛 `RtsError` 并携带
tick/actor/上下文定位，CLI 以结构化 JSON 输出、退出码 2，而不是崩溃栈。

## 5. 快照与恢复

快照带 `format_version` 与内容 `state_hash`。在线引擎默认每 10 tick 与末 tick 落盘；
加载时校验版本与哈希，不符即 `E_SNAPSHOT_STALE`。跳转 `jump_to(t)` 选择 ≤t 的最近快照
重建引擎游标，只补放 `(snapshot_tick, t]` 区间，不重放全部历史；重建完成后才整体替换
视图状态，界面永不展示半合成状态。GUI 使用独立 `gui_snapshots/`，与演示/离线路径的
`snapshots/` 互不污染。

## 6. 帧预算与降级

`ReplayPlayer` 每帧最多合成 `allowance * speed` 个 tick（speed∈{1,2,4,8}）。每帧记录
耗时、尝试/实际 tick 数、降级标志与码 `E_BUDGET_EXCEEDED`。超预算时：

1. `allowance = max(1, allowance // 2)`（固定减半规则）；
2. 该帧标记跳帧刷新；
3. 后续不超预算的帧每帧恢复 1 档，直到基准值。

降级只改变“每帧展示多少 tick”的节奏。已合成 tick、快照、账本、最终哈希均由确定性引擎
产出，测试以“紧张预算（含人为尖峰）”与“宽松预算”两条在线路径最终状态哈希相等来锁定
该不变量。为让预算统计本身可复算，演示使用 `ScriptedClock`：常规 tick 成本为预算一半、
每第 7 个 tick 为预算 6 倍的尖峰，确定性触发降级；真实 GUI 使用 `MonotonicClock`。

## 7. 在线/离线核对

离线一次扫描全流生成权威 `tick_hashes.jsonl`、`ledger.jsonl`、`tick_errors.jsonl`。在线
按帧跑完全程后，核对器逐 tick 比较完整状态哈希（不是资源总量），并对整条规范化账本再做
一次哈希比对；首个不一致 tick 输出该 tick 全部事件上下文与参与 actor。另含 RNG 漂移对照
与跳转等价校验。当前样本集结果：全部场景 `reconcile_ok=true`、`ledger_match=true`、
`jump_all_match=true`，且都能定位到非空的 `rng_drift_tick`。

## 8. 错误码覆盖

| 码 | 触发链路 | 类型 |
| --- | --- | --- |
| `E_SEQ_GAP` | actor 本地 seq 不连续（seq_gap_match 注入） | 软，定位 tick/actor/expected/actual |
| `E_RESOURCE_OVERFLOW` | 征收投影超过 resource_cap，截断 | 软 |
| `E_QUEUE_BLOCKED` | 并发上限/资源不足/无空闲单位 | 软 |
| `E_RNG_DRIFT` | 链版本变更后首个哈希漂移 tick | 核对器报告 |
| `E_BUDGET_EXCEEDED` | 帧超预算；或构造期预算参数非法 | 帧统计 + 硬校验 |
| `E_SNAPSHOT_STALE` | 快照版本不符/哈希被篡改/文件损坏 | 硬 |
| `E_RULE_RANGE` | 规则非正整数、unit_cost>cap、种子/队列参数非法 | 硬 |
| `E_INPUT_MALFORMED` | 缺字段、类型错、未知 action/actor、坏 JSON 行 | 硬，带行号 |

## 9. 实际验证结果（本机 Python 3.11 + NumPy 2.4.6 + PySide6 6.12.0，Linux 容器外本地运行）

- `python3 -m unittest discover -s tests`：34/34 通过。
- 固定演示序列两次运行：89 个产物文件全部逐字节相同（SHA-256 比对）。
- 5 个场景（正常、资源越界、同 tick 并发争用、倍速跳转、序号断裂）在线/离线逐 tick
  哈希一致、账本一致、跳转等价、RNG 漂移可定位。
- 预算降级在确定性尖峰下被触发（如 fast_jump_match 超预算 10 次、跳帧 10 次），紧张与
  宽松预算最终状态哈希一致。
- `QT_QPA_PLATFORM=offscreen` 下 GUI 冒烟通过：加载、单帧推进、暂停、倍速、跳转后
  资源曲线历史点数等于 tick 数。

## 10. Windows 使用

```powershell
py -m pip install -r requirements.txt
.\run_demo.bat        # 固定演示 + 测试
.\run_gui.bat         # 生成样本并启动交互界面
# 或手动：
py -m rts_replay --workspace workspace generate
py -m rts_replay --workspace workspace offline normal_match
py -m rts_replay --workspace workspace gui
py -m rts_replay --workspace workspace hashcheck
```

代码仅使用跨平台标准库能力与纯整数业务运算（金额/资源无浮点），路径全部经
`pathlib/os.path` 处理，Windows 与 Linux 产物字节一致。
