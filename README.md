# SpMM 增量诊断与热块失效传导工作台

本项目是纯 Python/NumPy 的 CSR SpMM 诊断应用，计算：

```text
C = alpha * A * B + beta * C
```

其中 `A` 为 `M x K` CSR 稀疏矩阵，`B` 为 `K x N` 稠密矩阵。应用支持行级编辑、行块缓存、精确失效、热块统计、单次重算预算、线程池并发、版本化快照、故障注入和增量/全量一致性验证。

项目完全离线，样本、配置、块大小候选、精度档位、预算、热块容量和注入清单均由固定种子生成。

## 快速开始

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[test]"

python -m pytest -q
python -m spmm_diag demo --seed 20261123 --output-dir artifacts --snapshot-dir artifacts/snapshots
```

运行 demo 会创建/整体替换 `artifacts/snapshots`，并写出：

- `artifacts/demo_report.json`：完整机器可读验收报告。
- `artifacts/diagnostic_report.md`：中文运行摘要。
- `artifacts/final_c.npy`：最终 `C` 的 `float32` 结果。
- `artifacts/snapshots/current/`：最新完整快照，含 `manifest.json` 与 `state.json`。

临时目录使用 `tempfile.gettempdir()`，快照目录由命令参数显式指定。运行时不需要 Docker、网络或外部数据。

### Linux / WSL

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest -q
python -m spmm_diag demo --seed 20261123 --output-dir artifacts --snapshot-dir artifacts/snapshots
```

## 目录结构

```text
spmm_diag/
  audit.py        热块事件流重放与预算不变量校验
  config.py       系数、分块、顺序、精度、并发、预算和热块容量配置
  csr.py          CSR 生成、三种分布、校验、指纹和行级编辑
  diagnostics.py  稳定诊断码与诊断排序
  engine.py       增量引擎、依赖失效、并发调度、全量重建和一致性比对
  fingerprint.py  规范化 JSON/NumPy 字节哈希
  fixture.py      固定种子演示样本与候选配置生成
  graph.py        输出行块依赖图
  hot.py          容量受限热块表与确定性逐出
  injections.py   四类独立故障注入
  kernels.py      fp32/定点 SpMM、分块、顺序变体、饱和溢出和行结束指纹
  snapshot.py     目录级原子快照、校验、过期检测和版本回退
  demo.py         完整验收场景
  cli.py          命令行入口
tests/            25 项自动化测试
artifacts/        已生成的验证报告、最终结果和快照
```

## 核心实现

### 1. CSR 构造与校验

- 支持 `uniform`、`powerlaw`、`extreme` 三种每行 NNZ 分布。
- 每行列索引使用无放回抽样后升序排序，保证行内严格递增。
- 校验偏移数组长度与单调性、偏移终点/NNZ 一致性、列范围、行内严格递增、数值有限性。
- CSR 结构错误输出 `E_CSR_INVALID`，非有限元素输出 `E_MATRIX_NONFINITE`。
- 结构指纹由分布类型、形状、行来源、行偏移、列索引和值内容派生 SHA-256。

### 2. 双路径 SpMM

两条路径都按输出行块计算，并支持行内 NNZ 分块与部分和二次累加。

`fp32` 路径：

- A、B、累加缓冲、alpha/beta 合并全程使用 `numpy.float32`。
- 每个 NNZ 对 B 对应行执行一次 axpy 式累加。

定点路径：

- A、B 按 `1 / 2^fractional_bits` 量化，采用远离零的四舍五入。
- 乘积装入声明位宽的带符号整数累加缓冲。
- 每次乘积加入和部分和合并均按饱和策略裁剪。
- 每个块记录按计算时序出现的首个溢出输出位置、方向和位宽。
- 累加结果除以 `2^(2*fractional_bits)` 反量化，再与 `alpha`、`beta*C` 按 fp32 合并。

`kernels.compare_float_and_fixed` 提供逐元素误差、元素数量和容差判断。演示中无饱和档位最大绝对误差为 `0.0172871`，容差为 `0.05`。

### 3. 分块、累加顺序与空行

- `row_block_size` 控制输出行块大小，`nnz_chunk_size` 控制单行 NNZ 分块大小。
- 边界块和边界 NNZ 块均精确裁剪。
- 支持：
  - `col_asc`：列索引升序。
  - `col_desc`：列索引降序。
  - `b_abs_desc`：按对应 B 行最大绝对值降序，列索引用于确定稳定次序。
- 每个块内先累加部分和，再按块序列做第二次累加。
- 零 NNZ 行输出 `empty_rows` 标记，并产生包含空行语义和依赖指纹的行结束指纹。
- 测试使用 `1, 2^24, -2^24` 证明 fp32 升序结果为 `0`、降序结果为 `1`；定点测试证明溢出方向会随顺序从 `+` 变为 `-`。

### 4. 行块依赖图与精确失效

每个输出块依赖记录以下内容：

- A 行集合及各 A 行指纹。
- 实际读取的 B 行集合及各 B 行指纹。
- 对应 C 行指纹，覆盖 `beta*C` 输入。
- alpha/beta、精度档位、分块和顺序指纹。
- 块输入指纹和块产出指纹。

失效规则：

- A 行替换：只失效包含该行的输出块。
- A 行删除：删除点所在块及其后因行号位移而变化的块失效。
- A 行追加：只失效新行落入的输出块。
- 行块大小变化：全部失效，并清理旧拓扑热块统计。
- NNZ 块大小、alpha/beta、精度或顺序变化：全部失效。
- B 行内容变化：只失效读取该 B 行的块。
- 未失效且依赖指纹完全一致的块直接复用，不进入计算线程池。

### 5. 热块与重算预算

- 热块表统计 `invalidations` 和 `recompute_nnz`。
- 展示顺序为失效次数降序、累计重算 NNZ 降序、块序号升序。
- 容量满时逐出失效次数最少者；并列时逐出块序号较小者。
- 每批按块序号稳定扫描待重算集合，并按块总 NNZ 扣减预算。
- 首个成本超过剩余预算的块及其后块保留在稳定待办队列。
- 未完成块输出 `E_RECOMPUTE_PENDING`，下一批继续按块序号处理。
- 热块变更、裁剪、容量调整和重算均记录为有序审计事件，可独立重放热块表。

### 6. 并发确定性

- 使用 `ThreadPoolExecutor` 计算彼此无依赖的已调度输出块。
- 工作线程数不超过配置 `max_workers`。
- 任务排序键统一为 `(block_id, stage)`。
- 预算在提交前按块序号扣减，结果按块序号收集，因此并发完成时序不会改变：
  - 输出块顺序。
  - 失效/复用清单。
  - 预算扣减序列。
  - 热块表。
  - 失败清单。
- 演示配置分别使用 1 和 4 个工作线程，8 份运行报告逐项一致。

### 7. 原子快照

- 先在快照根目录下写入独立临时目录。
- 写入 `state.json`，再写包含文件 SHA-256 的 `manifest.json`。
- 将旧 `current` 目录整体移动到 `versions/vNNNNNNNN`。
- 通过 `os.replace(stage, current)` 整体切换目录，不逐文件覆盖。
- 启动时校验 schema、文件集合、文件哈希、全局版本、配置指纹和输入指纹。
- 输入/配置/版本不一致输出 `E_SNAPSHOT_STALE`。
- 文件缺失、额外文件或校验和不一致输出 `E_SNAPSHOT_CORRUPT`。
- 当前快照无效时扫描最近完整版本，恢复仍满足依赖指纹的块，只重算被拒绝块。
- 快照包含输出块、依赖图、热块表、产出指纹、预算账本和热块事件流。

### 8. 故障注入

`run_injection_suite` 独立复制输入并分别注入：

- `INJ-01-CSR-COLUMN`：真实修改 `csr.col_indices[...]`，命中 `E_CSR_INVALID`。
- `INJ-02-NONFINITE-B`：真实修改 `B[row,column] = inf`，命中 `E_MATRIX_NONFINITE`。
- `INJ-03-ACC-WIDTH`：真实修改 `config.accumulator_bits = 0`，命中 `E_ACC_WIDTH_INVALID`。
- `INJ-04-BLOCK-SIZE`：真实修改 `config.row_block_size = 0`，命中 `E_BLOCK_SIZE_INVALID`。

每项记录注入编号、真实路径、修改前值、修改后值和失败清单；同时验证原始 CSR、B 和配置未被污染。

## 一致性口径

验收分两层，以避免把“最终状态”与“历史相关计数器”混为一谈：

1. **历史无关可观察量**：`C` 逐元素、行结束指纹、溢出记录、块产出指纹、依赖图。这些与从空白工作区按最终 `A/B/C/config` 一次性全量重建逐位一致。
2. **历史相关诊断量**：热块失效次数、累计重算 NNZ、预算扣减序列。它们依赖编辑批次和调度历史，无法由最终矩阵本身唯一反推。因此使用相同编辑/预算脚本的独立单线程重放逐项比对，并通过 `audit.py` 从有序热块事件流独立重建热块表、检查预算算术和每批块序号单调性。

若要求“只看最终 A/B/config 的一次性全量运行”也复现历史失效次数和预算序列，则必须把编辑/调度历史作为额外输入；这是信息本身不充分，而不是计算路径差异。当前实现显式保留该审计流，并支持快照持久化，因此不会伪造不可推导的历史值。

## 实际验证结果

环境：Python `3.11.0rc1`，NumPy `1.26.4`，临时目录可写。自动化测试共 25 项，全部通过。

固定种子 `20261123` 的演示样本：

- 形状：`M=30, K=18, N=6`。
- NNZ：`204`。
- 空行：`9`。
- 单行最大 NNZ：`18`。
- 最终块数：`10`。
- 工作线程：配置 `max_workers=4`；单线程对照 `max_workers=1`。

关键结果：

| 验收项 | 结果 |
| --- | --- |
| 增量 vs 全量逐元素一致 | `true` |
| 行结束指纹/溢出/块指纹/依赖图一致 | `true` |
| 并发 1 vs 4 的 8 份报告一致 | `true` |
| 热块审计流独立重放一致 | `true` |
| 行替换仅重算目标块 | `true`（块 2 重算，其余 14 块复用） |
| 零预算待办 | `true`（块 2 标记 `E_RECOMPUTE_PENDING`） |
| 前缀预算稳定扣减 | `true`（扣减块 0、1，剩余块待办） |
| fp32/定点可比较 | `true`（最大绝对误差 `0.0172871`） |
| 当前快照版本 4 有效 | `true` |
| 输入变化触发 `E_SNAPSHOT_STALE` | `true` |
| 快照回退仅重算受影响块 | `true`（块 1 被拒绝，其余块复用） |
| 配置变化触发 `E_SNAPSHOT_STALE` | `true` |
| 文件校验和损坏回退 | `true`（`E_SNAPSHOT_CORRUPT`，上一版本恢复） |
| 快照恢复后重建一致 | `true` |
| 四类故障注入全部命中且隔离 | `true` |

完整结果见 `artifacts/demo_report.json` 和 `artifacts/diagnostic_report.md`。

## 常用 API

```python
import numpy as np

from spmm_diag.csr import DistributionSpec, generate_csr
from spmm_diag.config import SpMMConfig
from spmm_diag.engine import IncrementalEngine, full_rebuild, compare_incremental_to_full

csr = generate_csr(30, 18, DistributionSpec("extreme", 0, 18), seed=7)
rng = np.random.default_rng(8)
b_matrix = rng.normal(size=(18, 6))
c_matrix = rng.normal(size=(30, 6))
config = SpMMConfig(alpha=0.75, beta=0.2, row_block_size=3, max_workers=4)
engine = IncrementalEngine(csr, b_matrix, c_matrix, config)
engine.run_until_complete()

engine.replace_row(4, [0, 3, 9], [1.2, -0.6, 0.9])
engine.run_until_complete()

reference = full_rebuild(engine.csr, engine.b_matrix, engine.c_matrix, engine.config)
assert compare_incremental_to_full(engine, reference)["all_exact"]
```

快照：

```python
from spmm_diag.snapshot import SnapshotStore, bootstrap_engine

store = SnapshotStore("artifacts/snapshots")
store.save(engine)
restored, info = bootstrap_engine(store, engine.csr, engine.b_matrix, engine.c_matrix, engine.config)
```
