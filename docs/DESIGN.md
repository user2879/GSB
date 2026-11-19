# 连续域约束求解与数值误差传播诊断 — 技术说明

## 1. 总体链路

```
YAML 规格 (spec.py 校验, sha256 spec_hash)
   │  固定 master seed
   ▼
连续域候选布局 (continuous.py)
   │  独立 PCG64 子种子: terrain/elevation/density/resources
   │  双精度高程场、密度场、地形抽样 uniforms、资源点
   ▼
量化 (quantize.py)
   │  q = clamp(round_dir(x/step)*step, 0, 1)，按区域分片
   │  逐图块 continuous/quantized/delta + 误差上界
   ▼
离散网格 (grid.py, grid_hash)
   ▼
约束判定 (constraints.py) + 误差区间 (util.classify)
   ▼
误差传播账本 (ledger.py, ledger_hash) + 边界报告
   ▼
独立复算器 (recompute.py) 仅消费工件，重算全部判定
```

## 2. 双精度数值模型与确定性

* 所有阈值、权重、连续场均为 IEEE-754 float64；YAML 解析后立即 `float()` 转换并拒绝
  非有限值（`leveldiag/spec.py`）。
* 随机源：每个用途一个 `numpy.random.Generator(PCG64)`，子种子由
  `sha256("seed|purpose[|r=region][|i=index]")[:8]` 大端转整数派生，
  派生树完整写入 `continuous/index.json -> seed_tree`，可逐条复算。
* 顺序确定性：区域按 id 排序、cell 按 `(x,y)` 字典序、地形抽样按行主序；
  权重求和按名称固定顺序；不使用线程、不依赖 dict 遍历顺序。
* 序列化确定性：统一 `json.dumps(sort_keys=True, separators=(",",":"))` + LF，
  numpy 类型先转 Python 原生类型，浮点用最短往返 repr。
  全部内容哈希 = 规范化字节的 sha256；manifest 记录每个文件逐字节哈希。

## 3. 三处误差源的传播口径

| 误差源 | 度量位置 | 上界 |
| --- | --- | --- |
| 连续→整数量化 | mapping 分片每图块 `delta = q - x` | nearest: `step/2 + 1 ULP(1.0)`；down/up: `step + 1 ULP` |
| 权重归一化 | `continuous.weight_normalization` | `|Σw − 1| + n·eps`（eps 为 float64 机器精度） |
| 阈值比较 | 每条账本条目 `total_band` | 声明 `comparison_tolerance + propagated_error_bound` |

比例类派生量（通道宽度比例、资源密度）额外计入一次双精度除法的 1 ULP 算术误差
（`arithmetic_error_bound`）。条目中同时保留 `quant_error_bound`、
`normalization_error`、`arithmetic_error_bound` 与合成的
`propagated_error_bound`，误差链可逐段复核。

## 4. 阈值比较的确定性口径（边界处理）

统一在 `leveldiag/util.py:classify`，账本与复算器共用同一语义：

* `band = comparison_tolerance + propagated_error_bound`，`d = observed − threshold`。
* `gt`：`d > band` PASS；`d < −band` FAIL；否则 UNCERTAIN。`lt` 对称。
* `ge/le`：仅在明确违约（`d < −band` / `d > band`）时 FAIL；阈值落入观测值的
  误差区间（含严格相等）一律 **UNCERTAIN，禁止静默四舍五入**。
* 离散结构量（BFS 连通 0/1、资源计数、地形类别相邻）误差带为 0，退化为纯数学
  比较，结论确定；其量化不确定性在相关连续量条目（通道比例、密度）上承载。
* 边界报告 `boundary_report.json` 列出所有 UNCERTAIN 或严格相等的判定，标明
  确定通过/确定失败/不确定、观测值、阈值与可信区间。

## 5. 约束条目与真值来源

| kind | 观测值 | 方向 | 主要真值来源 |
| --- | --- | --- | --- |
| `channel_width_ratio` | 可通行相邻边/总相邻边 | ge | mapping 高程分片、grid、规格阈值 |
| `connectivity` | BFS 是否覆盖区域并集（0/1） | ge | grid.tiles |
| `resource_count_interval` | 资源点数 | interval | continuous.resources |
| `resource_density_cap` | 资源数/区域单元数 | le | resources + grid |
| `cell_density_max` | 最大量化单元密度 | le | mapping 密度分片 |
| `forbidden_adjacent` | 违禁相邻对计数 | eq | grid.tiles + 归一化留痕 |
| `weight_normalization` | `|Σw−1|` | le | continuous.weight_normalization |

## 6. 增量重生成闭包

`leveldiag/incremental.py` 先做叶子路径 diff，再按字段表声明闭包并实际执行：

* 仅阈值/容差字段（如 `resources.density_cap`）→ 只重写
  `ledger.json`、`boundary_report.json`、`manifest.json`、`spec.canonical.json`，
  且只重算对应 kind 的条目；连续域、映射、网格文件哈希保持不变。
* `quantization.*` → 重写 mapping 全部分片/索引、grid、ledger，**连续域分片不动**。
* 结构性字段（seed、regions、points、weights 等）→ 全量闭包。

实际重写文件集合与声明闭包逐文件比对，不一致即 `E_CLOSURE_MISMATCH`；
非闭包账本条目若内容变化同样报错。

## 7. 独立复算器

`recompute.py` 不 import 生成器内存状态，只读取工件：

1. manifest 逐文件 sha256 校验（篡改 → `E_LEDGER_STALE`）；
2. 重算 spec/mapping/ledger 内容哈希并与账本引用比对（过期 → `E_LEDGER_STALE`）；
3. 仅用条目内观测值、容差、传播误差上界重算状态与可信区间；
4. 首个不一致 → `E_LEDGER_MISMATCH`，明细含观测值、阈值、方向、
   量化前后 continuous/quantized/delta 与真值来源。

## 8. 业务错误码

| 代码 | 触发场景 |
| --- | --- |
| `E_SPEC_INVALID` | 缺失字段、权重偏差超容忍、区域越界/重叠、出生目标重合或越界等（带 field_path） |
| `E_QUANT_BOUNDS` | 量化值越出 [0,1] 或实际 delta 超过声明误差上界 |
| `E_LEDGER_STALE` | 工件哈希与 manifest/账本引用不一致 |
| `E_LEDGER_MISMATCH` | 复算结论与生成器账本不一致 |
| `E_BOUNDARY_AMBIGUOUS` | 严格模式下存在 UNCERTAIN 判定（工件仍完整写出） |
| `E_CLOSURE_MISMATCH` | 增量实际重算集合与声明闭包不一致 |

## 9. Windows 运行方式

```bat
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m leveldiag.demo demo --out demo_out
.venv\Scripts\python -m leveldiag.demo generate examples\sample_spec.yaml --out out\lv
.venv\Scripts\python -m leveldiag.demo recompute out\lv
```

代码只使用标准库 + NumPy + PyYAML，路径全部走 `pathlib`，输出强制 LF 与
UTF-8，不依赖操作系统换行、大小写排序或外部网络。
