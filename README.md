# snapdiag：Windows 本地块存储快照与回滚诊断组件

`snapdiag` 从零实现一个完全本地、可复现、可异常注入的固定块卷快照与回滚引擎。运行时只依赖 Python 3.11+ 标准库（`sqlite3`、`pathlib`、`hashlib`、`tkinter` 等），不访问网络、不需要账号、不使用 Docker 或其他容器。

## 已交付能力

- 固定块卷强校验：受支持块大小、镜像精确长度、块号边界、精确十六进制块写入；几何错误统一抛 `E_GEOM` 并给出首个违规字段。
- 快照清单 `m1` / `m2` / `m3` 编解码、位图/摘要/引用/内容摘要校验及历史格式归一，补齐字段和补齐依据可追溯。
- 清单整体替换：一次事务删除并重建全部快照条目，不增量合并；旧清单保留为只读备份，稳定输出新增、删除、替换项。
- 写时复制：快照创建时保存不可变全量引用视图；后续块改写先在同一 SQLite 事务中置位位图并登记旧块摘要，再原位写卷。
- 双回滚路径：低比例走位图扫描，高比例走链回放加全量块比较；阈值边界固定选择 full，两种路径最终镜像、恢复块集合与结论逐位一致。
- 原子回滚：临时镜像、摘要预检、持久化操作标记、`os.replace` 原子切换、切换后逐块复核。
- 中断恢复：`prepared` 标记与临时区域在重启后按卷摘要恢复、回退或标记 unknown，要求显式重跑。
- 版本化审计：逐快照检查位图/引用、内容对象摘要、完整 checkpoint 视图、链到初始卷、清单与快照版本；旧审计结论不会被覆盖，版本过期报 `E_VERSION`。
- 三类异常注入：块数据损坏 `E_INTEGRITY`、清单截断 `E_FORMAT`、清单版本冲突 `E_VERSION`，并写入 JSON Lines 结构化日志。
- Tkinter 诊断界面：快照链、位图概览、路径选择、临时区域、审计、清单与最近事件按固定顺序渲染；无 Tk 的 CLI/测试环境可正常使用。
- 可重复演示：小卷、大卷、高变更比例三种本地样本，自动验证三版本归一、整体替换、双路径一致、中断恢复、异常注入/修复和审计版本。

## Windows 环境准备

在 PowerShell 中执行：

```powershell
py -3.11 --version
py -3.11 -c "import sqlite3, tkinter; print(sqlite3.sqlite_version, tkinter.TkVersion)"

# NumPy 仅作为环境规范中的可选项；本组件运行时不依赖它。
# 若组织要求安装，请只使用本地 wheel 目录，不访问网络源：
py -3.11 -m pip install --no-index --find-links .\wheels numpy

py -3.11 -m snapdiag doctor --workspace .
```

本组件的运行依赖只有标准库。`doctor` 会检查 Python、`sqlite3`、Tk、可选 NumPy、工作区/tmp/backups 写权限、临时目录和可用磁盘空间。`tkinter` 不可用时 CLI、测试和演示仍可运行，`gui` 会以 `E_UI` 明确报告原因。

> 当前验证环境为 WSL/Linux 上的 Python 3.11.0rc1、SQLite 3.37.2；代码只使用跨平台标准库接口，路径与原子替换使用 `pathlib`/`os.replace`。该环境未安装 `libtk8.6.so`，因此实际未启动图形窗口；GUI 之前的稳定视图模型已由无头测试覆盖。Windows Python 官方安装包自带 Tk，可直接运行 `python -m snapdiag gui ...`。

## 快速开始

```powershell
# 1. 环境自检
py -3.11 -m snapdiag doctor --workspace .

# 2. 生成三种本地样本并执行完整验收演示
py -3.11 -m snapdiag demo --workspace .

# 3. 运行测试（30 项）
py -3.11 -m pytest -q

# 4. 查看样本诊断视图
py -3.11 -m snapdiag show --session .\samples\generated\small --target checkpoint-a

# 5. 打开 Tk 界面（Windows Python 自带 Tk）
py -3.11 -m snapdiag gui --session .\samples\generated\small --target checkpoint-a
```

也可以使用 `scripts\run_demo.ps1`、`scripts\run_demo.bat` 或 `scripts\run_tests.ps1`。

## 会话清单

示例 `samples\small.session.json`：

```json
{
  "schema_version": 1,
  "volume": {
    "image": "volume.bin",
    "block_size": 512,
    "block_count": 16
  },
  "manifest_format": "m3",
  "rollback_threshold": 0.5,
  "snapshot_plan": ["baseline", "checkpoint-a", "checkpoint-b"],
  "rollback_target": "checkpoint-a",
  "steps": [
    {"action": "snapshot", "snapshot_id": "baseline", "parent_snapshot_id": null},
    {"action": "write", "block": 0, "fill_hex": "11"},
    {"action": "snapshot", "snapshot_id": "checkpoint-a", "parent_snapshot_id": "baseline"}
  ]
}
```

- `volume.image` 必须位于会话目录内；卷长度必须恰好为 `block_size * block_count`。
- 支持块大小：`512, 1024, 2048, 4096, 8192, 16384`。
- `steps` 按顺序执行：
  - `snapshot`：省略父引用时默认指向上一个快照；根快照显式传 `null`。
  - `write`：`hex` 必须恰好填满一个块，或使用可整除块大小的 `fill_hex` 确定性模式。
  - `audit`：执行一次版本化审计。
- 不提供 `steps` 时，先按 `snapshot_plan` 建链，再执行 `writes`。

## 清单格式

清单是一个整体文档，`snapshots` 为条目列表。规范字段如下：

| 版本 | 文档字段 | 条目新增字段 |
| --- | --- | --- |
| `m1` | `format_version`, `volume_block_count`, `block_size`, `snapshots` | `snapshot_id`, `volume_block_count`, `bitmap_length`, `bitmap`（可从 SQLite 补） |
| `m2` | 同 m1 | `digest_table`, `bitmap_checksum`, `snapshot_version` |
| `m3` | m2 加 `manifest_digest` | `parent_snapshot_id`, `block_refs`, `content_digest` |

- 位图采用 MSB-first：块 `i` 位于字节 `i // 8` 的第 `7 - (i % 8)` 位；尾部未使用位必须为 0。
- `bitmap_length` 必须等于 `ceil(volume_block_count / 8)`。
- `m2` 的 `digest_table` 项数、索引集合必须与已置位块完全一致；`bitmap_checksum` 是位图字节的 SHA-256。
- `m3` 的 `block_refs` 与 `digest_table` 索引集合一致，`content_digest` 覆盖归一后的条目内容，`manifest_digest` 覆盖归一后的完整 m3 文档。
- 未知版本、缺失字段、未知字段、长度不自洽、摘要数量/索引不一致、内容摘要不一致均抛 `E_FORMAT`，错误对象携带 `snapshot_id` 和首个 `field`。
- `m1` 缺少摘要/引用/父链/版本时，使用 SQLite 磁盘状态补齐；归一结果携带 `source_format`、`filled_fields`、`fill_basis`。补齐后三版本指纹一致。

完整规范和摘要算法见 [docs/FORMAT.md](docs/FORMAT.md)。

## 常用命令

```powershell
# 初始化
py -3.11 -m snapdiag init --session .\run\demo --block-size 4096 --block-count 1024 --threshold 0.25
py -3.11 -m snapdiag init --session .\run\demo --spec .\samples\large.session.json --run-steps

# 快照与块写
py -3.11 -m snapdiag create-snapshot --session .\run\demo --snapshot-id base
py -3.11 -m snapdiag create-snapshot --session .\run\demo --snapshot-id child --parent base
py -3.11 -m snapdiag write-block --session .\run\demo --block 7 --hex <恰好 block_size 字节的十六进制>

# 回滚、恢复、审计
py -3.11 -m snapdiag rollback --session .\run\demo --path auto base
py -3.11 -m snapdiag recover --session .\run\demo
py -3.11 -m snapdiag audit --session .\run\demo

# 清单归一、整体替换、只读备份
py -3.11 -m snapdiag normalize-manifest --session .\run\demo --file .\m1.json --format m3 --output .\normalized.m3.json
py -3.11 -m snapdiag import-manifest --session .\run\demo --file .\normalized.m3.json
py -3.11 -m snapdiag reconcile-manifest --session .\run\demo
py -3.11 -m snapdiag manifest-backups --session .\run\demo

# 三类异常注入及修复
py -3.11 -m snapdiag inject --session .\run\demo --kind block-corruption --target base
py -3.11 -m snapdiag inject --session .\run\demo --kind manifest-truncation
py -3.11 -m snapdiag inject --session .\run\demo --kind version-conflict
py -3.11 -m snapdiag repair --session .\run\demo --target base

# 启动参数注入后直接诊断/显示/启动 GUI
py -3.11 -m snapdiag show --session .\run\demo --inject version-conflict --target base
py -3.11 -m snapdiag gui --session .\run\demo --inject block-corruption --target base
```

所有命令输出稳定 JSON；业务异常同时以非零进程码退出，并追加到 `logs\diagnostics.jsonl`。

## 业务码

| 业务码 | 含义 | 退出码 |
| --- | --- | --- |
| `E_GEOM` | 块参数、镜像长度、块号或十六进制载荷非法 | 2 |
| `E_FORMAT` | 清单版本未知、JSON 截断、字段/长度/摘要不一致 | 3 |
| `E_CHAIN` | 父引用缺失、链成环、位图与引用集合不一致 | 4 |
| `E_INTEGRITY` | 引用块数据摘要或回滚后逐块复核失败 | 5 |
| `E_VERSION` | 清单版本冲突、持久化审计结论过期 | 6 |
| `E_STATE` | 中断操作待重跑、会话状态/文件系统不可用 | 7 |
| `E_EXISTS` | 同名快照重复创建 | 8 |
| `E_UI` | Tkinter 不可用或界面启动失败 | 9 |

## 回滚路径

给定目标快照：

```text
ratio = 已置位块数 / 卷块数
auto: ratio < threshold  -> bitmap
      ratio >= threshold -> full
```

- `bitmap`：按目标位图升序读取并恢复引用块。
- `full`：按快照链从新到旧回放撤销引用，再应用目标引用，并对全部块做比较统计。
- 边界故意采用稳定决胜规则：`ratio == threshold` 选 `full`。结果记录 `boundary_hit=true`、精确分数阈值、比例、快照数、链深和原因。
- 两条路径都报告相同的目标位图块集合，最终镜像 SHA-256 与快照 checkpoint 一致。

## 原子性与中断

1. 读取卷与旧块数据，校验目标引用块摘要。
2. 在 `tmp\` 生成完整结果镜像并 `fsync`。
3. 将 `prior_sha256`、`expected_sha256`、完整路径、每块期望摘要写入 SQLite `operations` 和 `tmp\<operation>.json`。
4. `os.replace(temp, volume)` 原子切换。
5. 逐块重算 SHA-256；全部通过后标记 `completed`。

重启后的 `recover` 根据卷摘要、持久化操作记录和临时标记决定：

- 卷已是目标摘要：完成收尾并逐块复核。
- 卷仍是旧摘要：丢弃临时镜像，标记 `incomplete_unknown`，要求显式 `--retry`。
- 两者都不是：标记 unknown，禁止静默继续。

## 架构与文件

```text
snapdiag/
  errors.py       业务码与结构化错误
  geometry.py     卷几何、块 I/O、确定性样本
  bitmap.py       位图与稳定概览
  manifest.py     m1/m2/m3 编解码、归一、摘要
  store.py        SQLite 会话、COW、回滚、恢复、审计
  injection.py    三类异常注入、修复、复检
  workflow.py     会话脚本执行与样本生成
  view.py         无头视图模型 + Tkinter 界面
  cli.py          命令行入口
  demo.py         可重复验收演示
  doctor.py       Windows/本地环境检查
tests/            30 项自动化测试
samples/          三份会话清单与预生成本地样本
docs/             格式、设计与实际验证记录
artifacts/        演示报告、测试输出、环境报告
scripts/          PowerShell/Batch 启动脚本
```

每个会话目录包含 `volume.bin`、`session.sqlite3`、`manifest.json`、内容寻址 `objects/`、只读 `backups/`、`tmp/` 和 `logs/`。SQLite 是快照元数据/审计的原子提交源；内容对象以 SHA-256 命名，并保留只读备份用于损坏恢复。

## 实际验收结果

最终本地运行结果：

- `python -m pytest -q`：`30 passed in 9.89s`。
- `python -m snapdiag demo --workspace .`：8 个验收阶段全部成功，耗时 15.742 秒。
- m1/m2/m3 归一指纹均为 `d3b1497c...87318c84`（完整值见报告）。
- 位图、全量、边界自动三条回滚最终卷 SHA-256 均为 `0cc92abe...b82b7e`；恢复块 `[0..7]` 完全一致。
- 注入观测码：`E_INTEGRITY`、`E_FORMAT`、`E_VERSION`；修复后 128 个块全部逐块复核通过，问题数 0。
- 审计 1 通过；状态变更后读取旧结论报 `E_VERSION`；新审计生成版本 2，历史版本保留。

详见 [docs/VALIDATION.md](docs/VALIDATION.md)、`artifacts/demo-report.json`、`artifacts/pytest-output.txt`、`artifacts/doctor-report.json`。

## 约束与说明

- 仅操作本地文件与本地进程，不使用外部服务、账号或容器。
- 默认面向单进程/单会话使用；SQLite 使用外键、事务和 busy timeout，但未实现多写者协调协议。
- 普通块写先提交 COW 撤销元数据，再原位写卷；若两者之间强制终止，撤销记录仍可安全恢复/回滚。回滚镜像本身严格先写临时区、校验后原子替换。
- 大卷路径为追求可诊断性会构建内存中的临时镜像；对于超大卷应改为流式分块临时文件，这是后续扩展点，不影响当前固定块样本验收。
