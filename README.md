# layerbench — 本地容器镜像分层构建与构建缓存诊断工作台

面向本地 Windows 环境的**多阶段分层构建 + 构建缓存诊断**原型，聚焦**并行调度与资源预算**。
不使用 Docker 或任何容器运行时、不访问网络、不依赖第三方包，仅使用 Python 3.11 标准库
（`tkinter`、`sqlite3`、`subprocess`、`hashlib`、`threading`、`argparse` 等）。

## 1. 目录结构

```
layerbench/            核心包（纯标准库）
  errors.py            统一业务码与异常
  digest.py            确定性哈希组合、稳定序列化、CRLF/CR -> LF 换行标准化
  manifest.py          JSON 清单数据模型与强校验（E_MANIFEST 整体拒绝）
  executor.py          跨平台本地命令执行（Windows cmd / POSIX sh）、超时两级终止
  layer.py             层摘要确定性计算（调度与重放共用的唯一规则来源）
  cache.py             跨构建缓存索引（双索引 + 确定性 LRU 淘汰）
  ledger.py            SQLite 账本（WAL、running 未提交边界、事件时间线）
  scheduler.py         双重预算并行就绪调度、失败传播、强制终止注入、续建
  report.py            确定性报告与预算审计（E_AUDIT 首个偏差定位）
  snapshot.py          会话快照、内容摘要、整体替换恢复、unknown 识别
  replay.py            恢复后重放校验（E_REPLAY）与三类异常注入
  gui.py               可选 Tkinter 诊断窗口（无显示环境优雅降级）
  app.py / cli.py      编排服务与命令行入口
samples/
  manifests/           正常/失败/超时/小容量/坏清单样本
  workspace/           待复制本地文件与确定性 Python 脚本
tests/                 16 个 unittest 用例，覆盖 8 项业务要求
run_demo.sh / .bat     POSIX 与 Windows 一键演示
workdir/               缓存索引、SQLite 账本、报告、快照的预留工作目录
```

## 2. 确定性层摘要规则（业务要求 2）

每条指令的摘要分两段，规则集中在 `layerbench/digest.py` 与 `layerbench/layer.py`：

- **precursor（前驱指纹，执行前可算）**
  `hash(type, 规范化 params, 稳定序输入文件摘要集合, 稳定序前驱层摘要集合)`
- **layer_digest（层摘要）**
  - `run`：`hash(precursor, 命令结果摘要)`，命令结果摘要
    `= hash(exit_code, 标准化换行后的 stdout 摘要)`；
  - `copy/env/label`：`hash(precursor, "no-command-result")`。

哈希组合采用显式「类型标签 + 长度前缀 + 内容」帧格式（`digest.combine`），
所有集合先按稳定序排序、所有 dict 用 `sort_keys` 序列化，因此结果与线程交错、
就绪顺序、缓存命中状态无关。Windows 与 POSIX 的 stdout 先统一为 LF 再摘要，
保证同一逻辑命令跨平台字节级一致。

> 缓存键设计说明：执行类指令在运行前无法知道退出码与 stdout，因此缓存维护两级键：
> `precursor -> layer_digest -> 记录`。命中时复用上次真实执行的命令结果，层摘要保持一致。

## 3. 双重预算并行调度（业务要求 3、4）

- 全局 `Semaphore(global_slots)` + 每阶段 `Semaphore(quota)`，任一时刻真实执行数
  与阶段执行数均不超限；预算耗尽只排队，记录单调时钟等待开始/结束，绝不绕过。
- 就绪队列按 `(阶段拓扑序, 指令 seq)` 全序弹出；缓存命中在调度线程内联完成、
  **不占槽位、不启动子进程**，天然满足“命中层未被执行”。
- 超时为显式两级机制：先 `terminate()`（Windows `TerminateProcess` / POSIX `SIGTERM`
  进程组），宽限后仍存活则 `kill()`（`SIGKILL`），并回收句柄；以 `E_TIMEOUT` 记录
  实际机制与残留状态，不使用“失控命令”措辞。
- 非零退出以 `E_COMMAND` 标记，下游未执行指令迁移为 `skipped` 并记录触发上游；
  成功/失败指令的层摘要都写入账本。

## 4. 缓存容量与稳定淘汰（业务要求 5）

缓存索引同时持久化到 `cache/cache_index.json` 与 SQLite（`evictions` 表）。
容量超限时按全序选择被逐出条目，**不依赖 dict/set 遍历顺序**：

```
min(last_used_seq 升序, refcount 升序, layer_digest 字典序)
```

每次淘汰记录被逐出层、规则与“放弃复用”原因。

## 5. 报告、审计与快照恢复（业务要求 6、7、8）

- 每次构建产出 `reports/<build_id>/report.full.json`（含等待/耗时等测量值）
  与 `report.canonical.json`（剔除全部墙钟/单调时钟字段与运行态 `cache_hit`/峰值，
  同输入重复构建**逐字节一致**，并计算 `canonical_digest`）。
- 审计独立从账本 slot 事件重建占用时间线，校验：占用不超限、命中层未执行、
  失败传播与跳过集合自洽、已提交指令必有层摘要；偏差以 `E_AUDIT` 定位首个
  阶段/seq/字段/依据。
- 快照为自包含只读归档（清单副本、缓存副本、检查点后的单文件账本、meta 内容摘要）。
  恢复时比较清单指纹与缓存索引摘要，不一致即**整体替换**当前清单/缓存/账本
  （旧状态归档到只读 `replaced_at_*`，绝不增量合并）。
- 强制终止（`--crash-at stage/seq`，用 `os._exit(137)` 模拟硬杀）后，仅承认
  `instructions` 中完整提交记录；`running` 残留标记 `unknown` 并通过 `resume` 重跑，
  不承诺生成退出报告。
- 重放按快照清单与预算重建 DAG，重推 precursor/层摘要/命中/业务码/槽位占用并与账本
  逐项比较，差异以 `E_REPLAY` 定位首个不一致字段。测量字段显式排除在比较之外。

## 6. 业务码

| 码 | 含义 |
| --- | --- |
| `E_MANIFEST` | 清单强校验失败，整体拒绝、不部分生效 |
| `E_TIMEOUT` | 单指令执行超时（含机制与残留说明） |
| `E_COMMAND` | 执行类指令非零退出，下游传播 skipped |
| `E_AUDIT` | 报告预算/一致性审计首个偏差 |
| `E_SNAPSHOT` | 快照内容摘要/版本/缓存摘要不匹配或损坏 |
| `E_REPLAY` | 恢复后重放与快照记录不一致 |
| `E_BUDGET` | 预算参数非法（如注入的零槽位） |

## 7. 命令用法

```bash
# 强校验
python -m layerbench.cli validate --manifest samples/manifests/sample_manifest.json \
  --workdir workdir/demo --source-root samples/workspace
# 一次构建
python -m layerbench.cli build --manifest <清单> --workdir <工作目录> --source-root <样本根>
# 两次构建并比对 canonical 逐字节一致
python -m layerbench.cli verify --manifest <清单> --workdir <工作目录> --source-root <样本根>
# 快照恢复（整体替换）与重放
python -m layerbench.cli restore --snapshot <快照目录> --manifest <清单> ...
python -m layerbench.cli replay  --snapshot <快照目录> --workdir <工作目录> --source-root <样本根>
# 强制终止与续建
python -m layerbench.cli build ... --crash-at builder/2
python -m layerbench.cli resume --build-id <id> --manifest <清单> ...
# 异常注入（只改输入文件/参数，不改源码）
python -m layerbench.cli inject --kind corrupt-manifest|cache-mismatch|zero-slots ...
# 可选 GUI（Windows 自带 Tkinter；无显示环境返回退出码 4 并提示）
python -m layerbench.cli gui --manifest <清单> --workdir <工作目录> --source-root <样本根>
```

## 8. Windows 适配要点

- 命令执行在 Windows 走 `cmd /c`，POSIX 走 `/bin/sh -c`；清单命令用 `${PYTHON}`
  占位符，执行前替换为当前解释器，同一份清单跨平台可移植。
- 复制源强制为工作区内相对路径，拒绝绝对路径/盘符与 `..` 越界。
- 超时在 Windows 用 `TerminateProcess` 终止直接派生进程句柄；脱离进程组的孤儿进程
  不确定性只影响环境清理，不参与层摘要。
- 演示前在 Windows 手工确认：`python -c "import tkinter; print('ok')"` 能打印 `ok`。

## 9. 实际验证结果（本仓库 Linux 沙箱，Python 3.11.0rc1）

```
$ python -m unittest discover -s tests
Ran 16 tests in ~5s — OK
```

覆盖：清单强校验与路径越界、重复构建 canonical 逐字节一致、层摘要跨构建稳定、
缓存全命中且不执行、全局/阶段预算不超限、确定性 LRU 淘汰、失败传播跳过、
超时机制与残留、快照自洽与篡改检测、缓存不匹配触发整体替换、三类注入被拒、
强制终止 unknown 识别与续建、续建快照重放一致。

> 本沙箱未安装 Tkinter，`gui` 子命令按设计返回退出码 4 并给出提示；
> Windows 标准 Python 3.11 自带 Tkinter，可直接显示窗口。
