# 冷热分层与迁移组件 — 技术设计说明

面向 Windows 本地环境的纯 Python 组件：SQLite 元数据账本 + 本地目录对象载体 +
JSON 清单驱动 + 固定整数种子的确定性内容。无 Docker、无外部服务，全部数据落在
工作区本地目录。

## 1. 目录与模块结构

```
tier_mover/
  errors.py       业务错误码：E_SPEC/E_CAPACITY/E_INTEGRITY/E_VERSION/E_REPLAY
  applog.py       结构化 JSONL 日志（线程安全）
  content.py      确定性内容生成（python 标准库后端 / 可选 numpy 后端）与摘要
  manifest.py     三份 JSON 清单的加载与强校验（全有或全无）
  heat.py         显式整数热度公式
  planner.py      确定性迁移计划推导、容量模拟、批次切分
  ledger.py       SQLite 账本（格式版本、链式摘要、事务、版本演进、stale 防护）
  engine.py       目录扫描真值、原子搬运、批次执行、崩溃注入点
  recovery.py     强制终止恢复、未知标注、rollback/rebuild 两种处置
  replay.py       恢复后一致性重放，首个分歧点 E_REPLAY
  faults.py       三类故障注入（仅改输入文件/参数，不改源码）
  workspace.py    工作区引导与样例清单生成
  demo.py         7 个固定场景的可重复演示
  cli.py          命令行应用
  gui.py          Tkinter 桌面应用（tkinter 懒加载）
  tests/          28 个 unittest 用例
  sample_workspace/  随仓样例：tiers.json / objects.json / seed.json
```

工作区运行时布局：

```
<workspace>/
  tiers.json  objects.json  seed.json      输入清单
  ledger.db (+ ledger.db-wal/shm)           SQLite 账本
  tiers/hot/<sha256>.bin  tiers/cold/...    按内容摘要命名的对象文件
  logs/events.jsonl                         结构化日志
```

## 2. 确定性内容与强校验（业务要求 1）

- 种子参数文件 `seed.json`：`{"seed": <非负整数>, "backend": "python|numpy"}`。
- 对象内容按 `object_seed = SHA 派生("seed|key")` 后由确定性 RNG 生成固定字节数；
  默认 `random.Random` 后端跨平台稳定，numpy 后端用 `PCG64` 固定种子，二者均可复现。
- 文件名与 `content_digest` 均为内容的 SHA-256。
- 强校验规则：`bool` 不视为 `int`；容量/成本为非负整数、对象字节数为正整数、
  时间戳/计数非负；层标识与对象键唯一；阈值为 0..1000 的千分位整数；
  摘要必须等于按种子重算结果。
- `ErrorCollector` 累积全部违规，但抛出时给出**首个违规字段与原因**，整体拒绝，
  不做任何部分装载（错误码 `E_SPEC`，`details.all_violations` 保留全量）。

## 3. 整数热度公式与计划推导（业务要求 3）

```
anchor_ms    = max(last_access_ms)
age_ms       = anchor_ms - last_access_ms
recency_rank = 按 last_access_ms 降序、key 升序的稳定 0 基名次
heat         = heat_weight_recency * age_ms
             + heat_weight_frequency * recency_rank      # 全整数，越小越热
```

计划两阶段，顺序完全由整数键决定：

1. 热层占用超过 `hot_watermark_permille/1000 * capacity` 时，按热度升序
   （冷者先迁）逐对象迁出，直到回落水位线；
2. 冷层占用高于 `cold_reclaim_permille` 目标时，按热度降序把较热对象晋升回热层。

每一步模拟容量，任何时刻不允许突破层容量；容量不足时按 `on_capacity`：
`reject` 抛 `E_CAPACITY`（列出层 id、被拒对象、余量与需求字节），
`rewrite` 稳定跳过该对象继续尝试后续候选。计划序列经规范 JSON 序列化后取
SHA-256 得 `plan_digest`，同一输入逐字节一致（测试 `test_plan_byte_for_byte_deterministic`）。

## 4. 真实搬运与原子切换（业务要求 2）

每个迁移步骤：

1. 写目标层临时文件 `.<digest>.<pid>.tmp`，`flush + fsync`；
2. 对临时文件重算 SHA-256，与清单摘要不符即删临时文件并抛 `E_INTEGRITY`；
3. `os.replace` 原子替换到目标路径，再删除源文件；
4. 失败路径清理临时文件，保证不会出现对象同时归属两层。

`scan_tier` 以目录扫描结果作为字节数真值；每个批次提交后 `reconcile_bytes`
比对目录字节数/文件数与账本，不一致即 `E_INTEGRITY`。

## 5. 账本、成本与分批事务（业务要求 4）

SQLite 表：`meta / spec_versions / tier_state / object_state / plans /
plan_steps / batches / migrations / progress / conclusions / audit_log`。

- `meta.format_version` 为整数账本格式版本（当前 3），打开即核对，不匹配抛 `E_VERSION`。
- 层状态整数记录 `used_bytes / free_bytes / object_count / cost`，
  成本 `cost = used_bytes * unit_cost`；单次搬运另记录 `move_cost = size * rw_weight`。
- 批次先以 `open` 状态**立即提交**落盘（崩溃后可识别），对象全部搬运成功后，
  在**单事务**内写迁移记录、更新对象归属、各层占用、进度游标，并推进链式摘要。
- 链式摘要：`chain_digest_n = SHA256(chain_digest_n-1 || canonical(批次载荷))`，
  `verify_chain` 可检测外部篡改。

## 6. 版本演进与陈旧防护（业务要求 5）

- 策略按规范摘要去重；清单任意参数变化注册新版本，版本号单调递增。
- 注册新版本时事务内把所有 `pending` 计划、`active` 结论、`open` 批次整体置
  `stale` 并记录 `superseded_ms`；`superseded_plan_fragments()` 稳定列出被作废片段。
- `current_view()` 只读取活动版本数据，stale 不参与当前视图；结论为整体替换
  （新结论插入前旧结论置 stale），不做增量合并。

## 7. 强制终止恢复（业务要求 6）

崩溃模型：`--crash-after-key <key>` 在指定对象完成搬运、批次提交前以退出码
137 终止进程（仅参数注入，不改源码）。重启后：

- 只承认 `committed` 迁移记录；`open` 批次标记 `aborted`。
- `RecoveryService.inspect()` 交叉比对账本归属与两层目录物理文件：
  一致计入 `committed_objects`，不一致计入 `unknown_objects` 并记录键。
- 两种处置：`rollback`（按摘要回退源层并校验）或 `rebuild`（按目标层重建归属，
  更新账本）。游离文件（账本无登记）直接 `E_INTEGRITY`。
- 恢复后以目录扫描真值重建层占用，并再次 `reconcile_bytes`。
- 组件不生成、也不声称能生成强制终止瞬间的“退出报告”，只输出恢复报告
  （已提交/未知/已回退/已重建计数）。

## 8. 重放校验（业务要求 7）

`ReplayService` 从初始全热层放置重新推导计划，按账本 `migrations` 顺序逐条比对
键、策略版本、摘要、方向、源/目标层、字节数，再逐步重算归属、占用与成本；
任何分歧立即抛 `E_REPLAY`，携带 `object_key / mismatch_field / basis /
expected / actual`，定位首个分歧点。策略版本或摘要不一致在前置校验即以
`E_VERSION`/`E_REPLAY` 报首点，而非笼统失败。

## 9. 故障注入（业务要求 8）

| 注入 | 实现（只改输入/参数） | 业务码 | 可恢复性 |
| --- | --- | --- | --- |
| 容量越界 | 改写 `tiers.json` 冷层容量为最小对象字节-1 | `E_CAPACITY` | 拒绝于搬运前，账本无改动 |
| 对象截断 | 截断热层某个 `.bin` 一半字节 | `E_INTEGRITY` | 账本可正常打开，可恢复/重建 |
| 版本冲突 | checkpoint 后复制账本，改副本 `format_version=999` | `E_VERSION` | 原账本不受影响 |

三类事件均写入 `logs/events.jsonl` 结构化日志。

## 10. GUI 与线程模型

`tier_mover/gui.py` 用 Tkinter/ttk 构建多标签页桌面应用；tkinter 采用函数内
懒加载，无头环境下 CLI 与测试不依赖它。耗时动作在后台 `threading.Thread`
执行，通过 `queue.Queue` 向主线程推送结构化事件，避免界面卡死。

## 11. Windows 使用方式

```powershell
python -m tier_mover.cli init .\workspace --seed 20261122
python -m tier_mover.cli plan .\workspace
python -m tier_mover.cli run  .\workspace
python -m tier_mover.cli status .\workspace
python -m tier_mover.cli replay .\workspace
python -m tier_mover.cli inject .\workspace capacity|integrity|version
python -m tier_mover.cli recover .\workspace --mode rollback|rebuild
python -m tier_mover.gui          # 图形界面
.\run_demo.ps1                    # 一键全流程演示
.\run_tests.ps1                   # 测试套件
```

路径全部使用 `pathlib`，文件写入显式 UTF-8、换行 `\n`，SQLite 用 WAL，
无 POSIX 专属调用，可直接在 Windows 官方 Python 3.11+ 运行。
