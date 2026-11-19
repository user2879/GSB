# Deterministic Constraint Level Workbench

面向关卡设计的本地约束程序化生成与可复现验证工作台。项目只使用 Python 标准库与 PyYAML；`numpy` 作为任务环境依赖安装并在报告中记录版本，但核心生成/校验逻辑不依赖数值库或线程调度。

核心能力：

- 严格解析 YAML 规格，并用稳定错误码拒绝非法输入。
- 使用 SHA-256 按用途派生子种子和自实现 SplitMix64 PRNG，生成结果逐字节可复现。
- 为出生点到目标点、每个必需区域记录实际 BFS 路径证据。
- 对禁止相邻图块仅做确定性局部重采样，并记录修复轨迹。
- 独立校验器只读取规格和序列化关卡，重新计算所有约束。
- 规格层、管线层、工件层异常注入均只通过配置生效，不改源码。
- 首个违规按行优先坐标定位，并输出可回放的最小反例。
- 固定序列演示生成结构化报告、逐构件哈希和整棵树哈希。

## Quick Start on Windows

要求 Python 3.11 或更高版本。

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

set PYTHONHASHSEED=0
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set VECLIB_MAXIMUM_THREADS=1

python run_demo.py --out artifacts\demo-1
python run_demo.py --out artifacts\demo-2
python -m level_workbench compare --a artifacts\demo-1 --b artifacts\demo-2
```

PowerShell 等价设置：

```powershell
$env:PYTHONHASHSEED="0"
$env:OMP_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:NUMEXPR_NUM_THREADS="1"
$env:VECLIB_MAXIMUM_THREADS="1"
python run_demo.py --out artifacts\demo-1
```

CLI 也会主动把 OpenMP/BLAS 线程变量固定为 `1`。实现本身不读取字典迭代顺序、进程哈希随机值或线程调度结果。

## CLI

```text
python -m level_workbench demo --out DIR
python -m level_workbench validate-spec --spec SPEC.yaml
python -m level_workbench generate --spec SPEC.yaml --out DIR [--inject INJECTION.yaml]
python -m level_workbench verify --spec SPEC.yaml --level LEVEL.json
python -m level_workbench replay-counterexample --spec SPEC.yaml --counterexample CE.json
python -m level_workbench compare --a DIR_A --b DIR_B
```

`generate` 在解析规格前不会创建输出目录；规格非法时不会留下网格文件。连通不可满足时返回 `E_CONNECT_UNSAT` 且不写关卡文件。约束修复失败时会保留失败候选，以便独立校验和最小反例定位。

## Specification Format

示例见 `artifacts/demo-*/specs/normal.yaml`。字段说明：

```yaml
seed: 20261119
grid: {width: 20, height: 14}
terrain:
  weights: {grass: 0.58, sand: 0.20, rock: 0.14, water: 0.08}
  blocked: [water]
regions:
  - {id: grove, bounds: [2, 2, 6, 5]}   # x0,y0,x1,y1，闭区间
resources:
  count: [10, 14]
  allowed_terrain: [grass, sand]
spawn: [1, 7]
target: [18, 7]
forbidden_adjacent: [[rock, water]]
fixed_cells: []                        # 可选；这些地形不会被雕刻或局部修复
```

- `terrain.weights` 必须非空且总和为 `1.0`，容差 `1e-9`。
- `regions` 支持 `bounds` 或显式 `tiles`。每个区域以其行优先最小图块作为连通锚点，证据中记录锚点、完整路径和雕刻图块。
- `resources.count` 是闭区间；资源不能被放在 `blocked` 地形、出生点或目标点上。
- `forbidden_adjacent` 表示地形类型对的无向四邻接禁止关系。
- `fixed_cells` 是不可变格；若它切断出生点和目标点/区域，会返回 `E_CONNECT_UNSAT`，不会静默放宽。

## Determinism Design

1. 规格先被规范化为排序后的 JSON 结构，哈希使用 UTF-8、键排序、紧凑分隔符和固定 LF 文件结尾。
2. 每个随机用途从 `SHA-256(master_seed | purpose | parts | offset)` 派生独立 64 位子种子。用途包括 `terrain`、`resources`、`spawn_neighborhood`、`forbidden_repair`、`resource_repair_*` 等。
3. 随机数使用项目内 SplitMix64 和拒绝采样，避免 Python `random` 的实现依赖及模偏差。
4. 地形名称、固定格、区域、资源坐标、证据和报告键全部显式排序；BFS 邻居顺序固定。
5. 每次运行记录所有子种子的用途、参量、偏移、十六进制值和主种子，并写入关卡元数据。
6. 网格同时生成规范文本与 SHA-256；关卡内容另计算不包含哈希字段的 `content_sha256`。

## Generation Pipeline

固定阶段顺序：

1. 地形加权采样。
2. 应用不可变固定格并规范化出生点/目标点。
3. 使用 `spawn_neighborhood` 子种子处理出生点邻域。
4. 对目标点与排序后的每个区域执行 BFS；若需雕刻，只改写非固定格，并记录完整路径。
5. 应用管线 `pre_repair_force_terrain`。
6. 禁止相邻局部修复：每轮只重采样冲突格邻域，使用 `repair_forbidden/round/x/y` 子种子，记录每次坐标、旧值、新值和前后违规。
7. 资源采样、管线资源计数覆盖与确定性局部增删修复。
8. 应用 `post_generation_force_terrain`，故意制造的违规会进入最终自检。
9. 生成器对全部约束做最终证据记录；失败候选保留。
10. 独立校验器重新解析工件并复核；runner 对比双方逐约束结论。

生成器证据覆盖 `grid`、`spawn`、`target`、`connectivity`、`resources`、`forbidden_adjacent`。校验器不读取生成器实例、子种子、路径缓存或修复状态。

## Injection Format

完整示例见 `artifacts/demo-*/injections/`。所有字段都可序列化：

```yaml
id: explicit-injection
spec_patches:
  - path: resources.count             # 支持 a.b 与 regions[0].bounds
    value: [300, 320]
pipeline:
  subseed_offsets: {terrain: 17}
  skip_local_repair: false
  skip_resource_repair: false
  pre_repair_force_terrain: []
  post_generation_force_terrain: []
  resource_count_override: null
artifact_tamper:
  force_terrain: []
  resource_count_override: null
```

- `spec_patches` 在解析规格前应用，用于区域越界、资源密度越界等规格层异常。
- `subseed_offsets` 只改变指定用途的子种子，便于验证用途隔离。
- `skip_local_repair` / `skip_resource_repair` 用于管线层跳过局部修复。
- `pre_repair_force_terrain` 用于让局部修复面对指定初始违规。
- `post_generation_force_terrain` 在正常修复之后强制违规，用于验证首个坐标与最小反例。
- `artifact_tamper` 在生成器证据形成后修改序列化候选，用于触发 `E_VERIFIER_MISMATCH`。

注入配置会以规范化形式写入报告、注入 YAML，并计算 `injection_sha256`。

## Verification and First Violation

独立校验器检查：

- 网格尺寸、每行长度、地形声明和网格哈希。
- 出生点/目标点越界或落在阻塞地形。
- 目标点与所有区域锚点的 BFS 可达性及路径证据。
- 资源数量区间、重复坐标、越界坐标、非法地形和端点重叠。
- 所有右邻/下邻边的禁止地形对。

违规按 `(y, x, 违规优先级)` 排序，因此第一个坐标始终是确定性行优先位置。每个违规包含 `code`、`coord`、`message` 和 `input_fragment`。生成器与校验器逐约束状态不一致时，顶层返回 `E_VERIFIER_MISMATCH`，并列出 `{constraint, generator, verifier}` 差异。

最小反例类型：

- 禁止相邻：两个图块赋值，删除任一元素后违规消失；可独立回放。
- 资源超量：恰好 `maximum + 1` 个资源，删除任一元素后回到合法区间。
- 资源不足：无法通过删除元素消除，明确标记 `minimizable: false`。
- 连通性：把违规归约为最小阻塞割集；仅保留该割集时仍断开，移除任一元素即出现路径。
- 重复资源：两个同坐标条目，删除任一元素后不再重复。

反例包含来源规格/关卡/网格哈希。`replay-counterexample` 会核对规格哈希并重新执行对应局部规则。演示会把每个失败样本的 `minimal_counterexample` 内嵌在 `report.json`；可单独抽出为 JSON 回放。

```bat
python -m level_workbench replay-counterexample ^
  --spec artifacts\demo-run-1\specs\normal.yaml ^
  --counterexample artifacts\demo-run-1\counterexamples\illegal_adjacent.counterexample.json
```

## Stable Error Codes

| Code | Meaning |
| --- | --- |
| `E_SPEC_FIELD` | 缺失字段、类型错误、非法枚举或 YAML 结构错误 |
| `E_SPEC_WEIGHT` | 地形权重为空、负数、非有限值或总和不为 1 |
| `E_SPEC_BOUNDS` | 坐标/区域越界、资源区间非法或资源地形阻塞 |
| `E_CONNECT_UNSAT` | 固定障碍导致出生点到目标点或区域不可达 |
| `E_REPAIR_EXHAUST` | 禁止相邻或资源局部修复耗尽，保留最后轨迹 |
| `E_RESOURCE_RANGE` | 资源计数仍越界（例如注入并跳过资源修复） |
| `E_GRID_INVALID` | 工件网格尺寸、行长度或地形非法 |
| `E_VERIFIER_MISMATCH` | 生成器与独立校验器逐约束结论不同 |
| `E_INJECTION` | 注入配置非法或无法应用 |
| `E_LEVEL_MALFORMED` | 校验器收到的关卡工件结构非法 |
| `E_IO` | CLI 文件读写错误 |

## Outputs

`demo --out DIR` 生成：

- `specs/`：本地合成基础规格。
- `effective_specs/`：应用规格补丁后的有效规格。
- `injections/`：规范化、可序列化注入配置。
- `levels/`：成功或可保留失败候选的规范化关卡 JSON。
- `grids/`：规范网格文本，每行空格分隔、末尾 LF。
- `counterexamples/`：可直接回放的最小反例 JSON。
- `report.json`：固定序列结构化报告，含逐约束证据、首违规、最小反例和分层哈希。
- `report.md`：便于人工检查的摘要。
- `hash_manifest.json`：所有输出文件 SHA-256 与整次运行 `run_sha256`。

报告包含 `effective_spec_sha256`、`injection_sha256`、`normalized_spec_sha256`、阶段哈希、约束证据哈希、`generator_evidence_sha256`、`verifier_sha256`、`level_sha256`、`grid_sha256`、`counterexample_sha256` 和 `first_violation_sha256`。

## Tests

```bat
python -m unittest discover -s tests -v
```

测试覆盖规格错误码、无部分文件、相同种子逐字节一致、子种子偏移、连通证据、禁止相邻修复、资源局部修复、跳过修复、独立校验、工件篡改 mismatch、行优先首违规、三类最小反例回放、双运行目录逐字节一致。

实际执行结果与环境记录见 `VALIDATION.md`。

## Layout

```text
level_workbench/
  spec.py            # YAML 解析与稳定规格错误
  seeds.py           # SHA-256 子种子 + SplitMix64
  generator.py       # 约束生成、雕刻、局部修复、证据
  verifier.py        # 独立校验器
  counterexample.py  # 最小反例构造与回放
  injection.py       # 可序列化注入配置
  runner.py          # 固定序列、mismatch、报告与清单
  samples.py         # 全本地合成样本
  cli.py             # 命令行入口
tests/               # 27 项 unittest
run_demo.py          # Windows 友好演示入口
```

项目不访问网络、不调用远程服务、不依赖游戏引擎或外部素材。规格和注入样本均由 `samples.py` 在本地构造。
