# 训练检查点与恢复诊断工作台（本地确定性）

一个**纯本地、无网络、无容器**的检查点账本/增量失效/快照恢复诊断应用。所有数据由固定
整数随机种子经 NumPy 确定性生成，状态保存在本地 SQLite（`ledger.db`）与分片二进制
（`shards/*.bin`）中。提供命令行与 Tkinter 诊断界面（无 Tk 时自动降级为稳定文本视图）。

## 1. 环境准备（Windows 本地）

- 安装 Python 3.10+（安装时勾选 *tcl/tk and IDLE*，官方安装包默认含 tkinter）。
- 校验依赖：

```bat
python --version
python -c "import numpy, sqlite3, tkinter; print('deps ok')"
```

仅依赖 NumPy；`sqlite3`、`tkinter` 为标准库。无需 Docker、无需外部账号、不访问网络。

> 当前开发/验证机为 Linux，缺少 `libtk8.6.so`，因此 GUI 自动降级为文本视图；在标准
> Windows Python 上 `python -m workbench.cli status --workdir <dir> --gui` 会弹出 Tk 窗口。

## 2. 目录结构

```
ckpt_workbench/
  workbench/            应用包
    model.py            确定性模型、分片规则、分片 blob 与各类摘要
    hashing.py          规范化编码与 sha256 摘要（跨运行逐位一致）
    dependencies.py     有向依赖图、引用完整性、失效传播与拓扑序
    ledger.py           SQLite 账本、强校验、异常注入接口
    changeset.py        变更集、增量提交、幂等、崩溃点、全量重算
    verifier.py         增量结果 vs 全量重算（E_INCR 定位）
    snapshot.py         会话快照、版本识别、整体替换、只读备份
    recovery.py         强制终止恢复、未知分片、两种处置、重放（E_REPLAY）
    inject.py           三类异常注入（E_INTEGRITY/E_CHAIN/E_SNAPSHOT）
    gui.py              Tk 界面与文本降级，渲染顺序稳定
    demo.py             五个隔离场景的可重复演示
    cli.py              命令行入口
  samples/              本地确定性样本声明与变更集
  tests/                unittest 测试（12 个）
  run_demo.bat / run_tests.bat   Windows 一键脚本
```

## 3. 确定性样本（samples/）

`declaration.json` 声明整数种子、分片数、分片方式、模型维度与步数：

```json
{ "seed": 20261123, "shard_count": 2, "partition_mode": "layer",
  "layer_dims": [8, 6, 4], "steps": 2 }
```

分片方式（显式规则）：

- `layer`：每层一个分片，含 `W_l`、`b_l` 及各自动量（本例 2 个分片）。
- `unit`：每张量一个分片（本例 4 个分片）。切换方式会重划所有分片边界。

变更集示例：`change_train.json`（推进训练）、`change_partition.json`（layer→unit）、
`change_cursor.json`（仅改数据游标）。支持的 op：`train_steps`、`set_partition_mode`、
`set_cursor`、`set_rng_state`、`drop_tensor`。

## 4. 快速开始

```bash
# Linux/macOS（Windows 用等价的反斜杠路径，或直接运行 run_demo.bat）
python -m workbench.cli init     --workdir out/ws --decl samples/declaration.json
python -m workbench.cli submit   --workdir out/ws --decl samples/declaration.json --change samples/change_train.json
python -m workbench.cli verify   --workdir out/ws --decl samples/declaration.json --changes samples/change_train.json
python -m workbench.cli status   --workdir out/ws                 # 文本诊断视图
python -m workbench.cli status   --workdir out/ws --gui           # Windows 弹出 Tk 界面
python -m workbench.cli replay   --workdir out/ws --decl samples/declaration.json
```

一键完整演示（五个互不污染的独立副本）：

```bash
python -m workbench.cli demo --root out/demo --decl samples/declaration.json \
  --changes samples/change_train.json samples/change_partition.json samples/change_cursor.json
# Windows: run_demo.bat
```

## 5. 业务规则与错误码对照

| 需求 | 实现位置 | 错误码 |
|---|---|---|
| 确定性构造与摘要登记 | `model.py`, `ledger.py` | — |
| 账本强校验（唯一/父引用无环/分片集合/版本单调/内容摘要） | `ledger.py:Ledger.validate` | `E_LEDGER` / `E_CHAIN` / `E_INTEGRITY` |
| 依赖建模与失效传播、删除被引用分片 | `dependencies.py`, `changeset.py` | `E_DEPEND` |
| 仅重写受影响分片、未受影响逐字节不变 | `changeset.py`, `verifier.py` | `E_INCR` |
| 单事务原子提交、幂等、版本串行化 | `changeset.py`, `app.py` | — |
| 快照版本识别、整体替换、只读备份 | `snapshot.py` | `E_SNAPSHOT` |
| 强制终止恢复、未知分片、回退/重跑、重放 | `recovery.py`, `changeset.py` | `E_REPLAY` |
| 三类异常注入 | `inject.py` | `E_INTEGRITY` / `E_CHAIN` / `E_SNAPSHOT` |

说明：父引用悬空单独以 `E_CHAIN` 报告；其余结构性违例（重复 id、分片数不一致、版本号
非单调等）以 `E_LEDGER` 整体拒绝并稳定列出每条违例；摘要/分片字节不匹配以 `E_INTEGRITY`
报告。所有错误均带稳定 code 与可读 details，定位到检查点/分片/字段。

## 6. 关键设计

- **规范化摘要**：`hashing.canon_bytes` 对类型加标签、整数定长帧、NumPy 数组取小端原始
  字节，避免平台相关浮点 `repr`；同参数组合多次运行得到逐位一致摘要。
- **依赖图**：源节点 `param:* / momentum:* / part:* / rng / cursor` →
  `shard:* → partition/optimizer → content`。失效沿边前向传播，输出稳定拓扑序与每步依据。
- **增量正确性**：仅受影响分片重算 blob 并将 `shard_version+1`；未受影响分片复用父检查点
  的 blob 路径（逐字节同一文件）且版本号不变。`verifier` 用从变更集全量重算的结果逐项
  比对，首个不一致即抛 `E_INCR`。
- **原子性**：单次 `BEGIN IMMEDIATE ... COMMIT` 内写分片行、派生摘要、账本版本、进度游标；
  WAL 模式下提交前其他读取者不可见、提交后整体可见。重复变更集以变更指纹去重，幂等不重复。
- **快照**：携带整数 `format_version` 与内容摘要；重新打开先比版本与摘要，不一致即**整体
  替换**（非增量合并），旧状态复制为 `backup_*` 只读目录并稳定列出被替换检查点集合。
- **强制终止**：`--crash-point before_commit` 模拟分片 blob 已写、事务未提交。SQLite 自动
  回滚事务，仅保留在途标记与未提交 blob；重启后这些分片标记为 `UNKNOWN`，提供
  `recover --mode rollback`（回到已提交父检查点）或 `--mode rerun`（按变更集重跑）。
  系统**不虚构**强制终止时的退出报告，只信任已完整提交的记录。

## 7. 异常注入用法（独立副本，避免污染）

```bash
python -m workbench.cli inject --workdir <ws> --kind shard_corruption \
  --checkpoint cp_baseline --shard shard000      # 期望 E_INTEGRITY
python -m workbench.cli inject --workdir <ws> --kind dangling_parent \
  --checkpoint cp_baseline                        # 期望 E_CHAIN
python -m workbench.cli inject --kind snapshot_mismatch --snapshot <snapdir>  # 期望 E_SNAPSHOT
```

## 8. 测试与实际验证结果

```bash
python -m unittest discover -s tests -v   # 或 Windows: run_tests.bat
```

本机（Python 3.11.0rc1, NumPy 1.26.4, SQLite 3.37.2）实测：

- 12 个 unittest 全部通过。
- `demo` 五场景全部 `status=OK`，结论见 `out/demo_report.json`：
  - 增量：游标变更下 4 个未受影响分片 blob 逐字节一致且 `shard_version` 不变；独立镜像
    重跑整条链，摘要签名完全一致（确定性）。
  - 快照：版本一致→差异为空；制造分歧后检出 3 项版本差异；整体替换后回到一致，旧状态
    存为只读 `backup_*`。
  - 崩溃：训练类变更在途时 2 个分片标记 UNKNOWN；回退与重跑两种处置后在途标记均清除，
    重放 `status=OK`。
  - 异常注入：分别稳定报 `E_INTEGRITY`、`E_CHAIN`、`E_SNAPSHOT`。
  - 幂等/并发：重复提交被跳过、不产生重复条目；不同变更集按账本版本单调串行化，每次提交
    读到的父引用均为已提交检查点。
