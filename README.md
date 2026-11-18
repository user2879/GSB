# tier_mover — 确定性冷热分层与迁移组件

面向 Windows / 本地环境的冷热分层领域组件。以本地目录中的对象文件作为数据载体，
以本地 SQLite 文件（标准库 `sqlite3`，无需数据库服务）作为元数据账本，通过本地
JSON 清单声明分层策略，按显式整数热度公式推导迁移计划，在热层与冷层目录之间真实、
原子地搬运对象，并把每一次迁移的前后占用、对象归属、原因与摘要写入账本。

同一份输入与同一注入序列，会得到**逐字节一致**的迁移计划与账本结论（见“确定性”）。

## 1. 环境准备

- Windows 本地安装 Python 3.10+，确认 `python --version` 与 `pip --version` 可用。
- 安装 NumPy（用于确定性样本内容生成与摘要）：
  `python -m pip install -r requirements.txt`
- SQLite 由 Python 标准库 `sqlite3` 提供，无需额外服务。
- 不使用 Docker，不访问网络或外部存储（除安装依赖外，运行期全部落在工作区本地）。

> 本次实现实际验证环境为 Python 3.11 + NumPy 2.4.6 + SQLite 3.37.2（Linux 容器内
> 开发验证）。代码仅使用跨平台标准库能力（`pathlib`、`os.replace`、`tempfile`、
> `sqlite3`），在 Windows 上行为一致。

## 2. 目录结构

```
tier_mover/            可集成 Python 包
  errors.py            稳定业务码与结构化异常（E_SPEC/E_INTEGRITY/E_CAPACITY/E_VERSION/E_REPLAY）
  models.py            Tier/Obj/Rule/Manifest 不可变领域模型
  heat.py              显式整数热度公式与全序排序
  manifest.py          JSON 清单加载与强校验（E_SPEC，含字段路径）
  samplegen.py         固定 NumPy 种子的确定性样本生成
  storage.py           摘要校验、临时文件+原子替换搬运、双归属检测
  planner.py           纯函数迁移计划推导（放置/降级/晋升/推迟/保留）
  ledger.py            SQLite 账本、版本识别、整体替换快照、每对象事务、结论导出
  engine.py            编排：物化→计划→搬运→提交→校验→恢复
  replay.py            恢复后一致性重放校验（首个不一致即 E_REPLAY）
  faults.py            异常注入配置（只改输入文件/参数，不改源码）
  logging_util.py      确定性 JSONL 结构化日志（逻辑时钟，无墙钟时间）
  cli.py               命令行入口
tests/                 unittest 套件（18 个用例）
examples/run_demo.py   三次会话 + 三类注入 + 中断恢复的完整演示
run_tests.py           测试入口
pyproject.toml         打包元数据与 console_script
requirements.txt       依赖（numpy）
```

## 3. 关键设计

### 3.1 热度公式（纯整数、全序、确定性）

```
HEAT_TIME_SCALE = 2**40
heat = access_count * HEAT_TIME_SCALE + (last_access_ms // 1000)
```

访问计数为主导项，秒级时间戳只在计数内打破平局；剩余平局再以 `(key, digest)`
收尾，构成与字典/哈希遍历顺序无关的全序。排序辅助见 `tier_mover/heat.py`。

### 3.2 强校验（E_SPEC）

`tier_mover/manifest.py` 聚合所有问题后一次性拒绝，错误项带层/对象键与 JSON 字段路径，
覆盖：层标识唯一、容量与单位成本非负整数、布尔值不被当作整数、对象键唯一、字节数为
正整数、访问时间戳/计数非负、`hot_watermark_bytes <= cold_reclaim_bytes`、批量大小为
正整数，以及阈值不得超过对应层容量。

### 3.3 原子搬运（E_INTEGRITY）

文件以内容摘要命名。迁移流程（`storage.transfer_object`）：

1. 在**目标层目录内**写唯一 `.tmp` 临时文件并 fsync；
2. 校验临时文件 sha256 == 声明摘要，不符抛 `E_INTEGRITY`（含对象键与两层/期望摘要）；
3. `os.replace` 原子替换到目标最终路径（Windows/Linux 同盘原子）；
4. 目标 durable 后才删除源文件。

崩溃窗口被显式覆盖：若在第 3、4 步之间被杀，对象会短暂双归属；恢复时按摘要重验、
保留目标并删除冗余源，绝不伪造完成报告。

### 3.4 计划推导与容量约束

`planner.derive_initial_plan` 做首次热优先放置；`derive_cycle_plan` 基于当前归属：

- 热占用 > 热水位：按 `(evict_priority, heat, key, digest)` 最冷优先降级；
- 冷占用 > 冷回收阈值：最热优先晋升，且晋升后热层不得越过水位/硬容量；
- 目标层余量不足时**稳定推迟**并记录原因（`postponed`），保证任意时刻占用不超容量；
- 计划输出每个对象的方向、原因、源层、目标层、摘要与热度，全部显式排序。

### 3.5 容量与成本账本（E_CAPACITY）

`ledger.py` 逐层以整数记录 `used_bytes / free_bytes / object_count / storage_cost`
（存储成本 = 占用字节 × 单位成本），迁移成本 = 字节 × 单位成本 ×
（源读权重 + 目标写权重），可在 `object_cost_breakdown` 中逐对象回溯。每次对象迁移在
**同一 SQLite 事务**内写迁移记录、两层占用与进度游标；提交后引擎再校验账本占用与目录
实际字节逐项一致，超容量抛 `E_CAPACITY`。

### 3.6 快照版本识别与整体替换（E_VERSION）

账本持久化策略版本、层参数版本、对象归属、各层占用与迁移轨迹。重开时
`check_versions` 比较快照版本与会话版本，不一致抛 `E_VERSION`；`replace_snapshot`
把旧库复制到 `snapshots/` 并置为只读（`0o444`），再以当前清单**整体替换**层参数、
对象归属与账本状态（绝不增量合并），`diff_versions` 稳定列出变化的层与对象及字段。

### 3.7 中断恢复与重放（E_REPLAY）

- 每对象迁移前先以独立事务标记 `in_flight` 与未知游标；物理切换与完成事务分离。
- 强制终止（`--inject kill`，用 `os._exit` 模拟 SIGKILL/任务管理器结束）后重启，
  `Engine.recover` 只信任已完整提交的记录；对 in-flight/双归属对象重验摘要后解析，
  回滚未完成的非原子周期，按校正后的物理占用与确定性计划重跑剩余迁移，并写
  `recovery_report.json`。不会生成“强制终止那一刻”的退出报告。
- `replay.replay` 按记录的策略版本与注入序列重推计划、重放已完成迁移，逐项比对归属、
  方向、各层占用与摘要；首个不一致即 `E_REPLAY`，定位到对象键、层标识与字段并给出依据。
  策略版本或对象摘要变化同样以 `E_REPLAY` 说明不可复现原因。

### 3.8 异常注入（只改输入/参数）

| 注入 | 手段（不改源码） | 业务码 |
| --- | --- | --- |
| `capacity` | 参数把热层容量覆盖为 1 字节 | `E_CAPACITY` |
| `truncate` | 截断暂存区某个对象输入文件 | `E_INTEGRITY` |
| `version` | 参数传入不匹配的策略/层版本 | `E_VERSION` |
| `kill` | 目标切换后、源删除前硬终止进程 | 退出码 9，进入恢复流程 |

所有异常与关键事件都写入确定性 JSONL 结构化日志（`logs/*.jsonl`）。

## 4. 命令行用法

```bash
# 生成确定性样本（manifest.json + staging/ 摘要命名文件）
python -m tier_mover.cli sample <workdir>

# 首次分层放置 / 迁移周期
python -m tier_mover.cli initial <workdir> --manifest <workdir>/manifest.json
python -m tier_mover.cli cycle   <workdir> --manifest <workdir>/manifest.json

# 中断恢复 / 一致性重放 / 导出账本结论
python -m tier_mover.cli recover     <workdir> --manifest <workdir>/manifest.json
python -m tier_mover.cli replay      <workdir> --manifest <workdir>/manifest.json
python -m tier_mover.cli conclusions <workdir>

# 版本识别 + 整体替换快照
python -m tier_mover.cli replace <workdir> --manifest <new-manifest.json> --force

# 三类异常注入（业务码经 stderr 的 JSON 返回，CLI 退出码 2）
python -m tier_mover.cli initial <w> --manifest <w>/manifest.json --inject capacity --capacity-override 1
python -m tier_mover.cli initial <w> --manifest <w>/manifest.json --inject truncate --target-key obj-03
python -m tier_mover.cli cycle   <w> --manifest <w>/manifest.json --inject version \
    --policy-version-override policy-v9 --tier-param-version-override tier-v9
python -m tier_mover.cli cycle   <w> --manifest <w>/manifest.json --inject kill --kill-after-commit 1
```

Windows 下可直接运行演示：`examples\run_demo.bat`（等价 `python examples/run_demo.py`）。

## 5. 一键演示与稳定输出

```bash
python examples/run_demo.py     # 三次会话 + 三类注入 + kill 恢复 + 确定性比对
python run_tests.py             # 18 个 unittest 用例
```

演示把产物写到 `artifacts/demo/`，稳定顺序汇总在 `artifacts/demo/demo_summary.json`：

- session1：首次分层放置 + 首个迁移周期；
- session2：容量收紧、版本升级 → 只读归档旧快照 + 整体替换 + 重迁移；
- session3：迁移中途硬终止（退出码 9）→ 恢复（解析双归属未知对象并重跑）→ 重放通过；
- determinism：两次独立运行的计划与账本结论 sha256 全部相等；
- fault_injections：`capacity→E_CAPACITY`、`truncate→E_INTEGRITY`、`version→E_VERSION`。

## 6. 实际验证结果

本工作区已实际执行并通过：

- `python run_tests.py` → `Ran 18 tests ... OK`（清单校验、计划/容量、端到端、
  三类注入、kill→recover→replay、重放不一致检测、逐字节确定性）。
- `python examples/run_demo.py` → 退出码 0；三次会话重放均 `status=ok`，
  session3 `kill_exit_code=9` 且恢复后重放通过，确定性四项全为 `true`，
  三类注入业务码依次为 `E_CAPACITY / E_INTEGRITY / E_VERSION`。
- 两次独立同输入运行的 `initial_plan.json`、`cycle_plan.json`、
  `*_conclusions.json` 与 JSONL 日志 sha256 完全一致（逐字节确定性）。

## 7. 作为库集成

```python
from tier_mover import Engine, JsonlLogger, load_manifest, build_sample, replay

build_sample("./data")
manifest = load_manifest("./data/manifest.json")
with JsonlLogger("./data/logs/app.jsonl") as log:
    engine = Engine("./data", manifest, log)
    engine.run_initial()
    engine.run_cycle()
print(replay("./data", manifest)["status"])  # "ok"
```
