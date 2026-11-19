# GSB 本地等值联接规模估算与采样诊断实验台

本项目是一个完全离线的 Windows 友好实验台，用本地 JSON 声明两张数值基表，生成确定性 NumPy 列式样本，计算等深分位直方图与确定性样本账本，并同时给出：

- 直方图桶重叠路径的联接行数估算；
- 抽样键频次重标定路径的联接行数估算；
- NumPy 真实等值联接结果；
- 显式容差区间及 `E_EST` 越界诊断；
- 逐重叠桶对的双路径贡献差、稳定排序和前 K 归因；
- 参数版本整体替换、旧结论隔离；
- 会话快照、恢复重算、逐桶重放比较；
- 样本损坏、参数版本缺失、分桶数为零三类可注入异常。

所有运行只使用本地文件系统与本地进程，不使用 Docker、容器、线上数据源或外部账号。

## 快速开始

### Windows 环境

1. 安装 Python 3.10 或更高版本。使用 Windows 官方安装包时保留默认的 Tcl/Tk 组件。
2. 打开 PowerShell 或命令提示符，在项目目录执行：

```powershell
py -3 -m pip install -r requirements.txt
py -3 -m unittest discover -s tests -v
py -3 scripts\verify_all.py
```

3. 启动 Tkinter 实验台：

```powershell
py -3 run_experiment.py gui
```

也可以直接双击 `run_gui.pyw`。命令行方式：

```powershell
py -3 run_experiment.py run --spec examples\demo_skewed.json --workdir var\demo --report reports\demo_full_report.json
```

### 当前 Linux 验证环境

本次交付在 Python 3.11.0rc1、NumPy 2.4.6 上完成核心与 CLI 验证。该执行环境的 Python 未安装 `tkinter`，因此 GUI 完成了语法编译检查，但未进行交互式窗口验证；Windows 官方 Python 默认包含 Tkinter。核心引擎不依赖 GUI。

## 项目结构

```text
gsb/
  model.py            场景模型、强校验、E_SPEC/E_VERSION/E_BUCKET
  generation.py       确定性列式生成、NPZ、摘要清单与损坏检测
  filtering.py        谓词过滤与空值语义
  histogram.py        等深分位桶、显式边界、桶摘要、重叠桶对
  sampling.py         物理顺序无关抽样、键频次账本、抽样估算
  truth.py            NumPy 精确等值联接与真实桶贡献
  estimation.py       定点估算、路径选择、容差区间、主导偏差桶
  attribution.py      逐桶对差异归因与稳定分类
  session.py          会话编排、参数演进、整体替换、陈旧隔离
  snapshot.py         快照保存、恢复重算与 E_REPLAY
  logging_store.py    SQLite + JSONL 结构化日志
  cli.py              命令行入口
  gui.py              Tkinter 本地操作台
examples/
  demo_skewed.json    整数倾斜场景
  demo_float.json     浮点低基数、含步长的场景
  demo_estimate_miss.json  连续高基数浮点的 E_EST 场景
  anomalies/          由脚本生成的异常夹具
scripts/
  make_demo.py        生成异常夹具与损坏 NPZ
  verify_all.py       一键验证并写出 reports/verification.json
tests/
  test_gsb.py         15 项端到端与确定性测试
reports/
  verification.json   最近一次实际验证记录
```

运行产生的样本、快照与日志默认位于 `var/`，已在 `.gitignore` 中排除。

## 场景 JSON

```json
{
  "schema_version": 1,
  "name": "skewed-orders-customers",
  "tables": {
    "customers": {
      "declared_rows": 1200,
      "seed": 7001,
      "skewed": false,
      "columns": [
        {"name": "id", "type": "int", "unique": true, "domain": 1200},
        {"name": "active", "type": "int", "domain": 2}
      ]
    },
    "orders": {
      "declared_rows": 6000,
      "seed": 7002,
      "skewed": true,
      "skew": {"hot_fraction": 0.82, "hot_domain_fraction": 0.03},
      "columns": [
        {"name": "customer_id", "type": "int", "domain": 800},
        {"name": "amount", "type": "int", "domain": 24000}
      ]
    }
  },
  "join": {
    "left_table": "orders",
    "right_table": "customers",
    "left_key": "customer_id",
    "right_key": "id",
    "left_predicates": [{"column": "amount", "op": "ge", "value": 12000}],
    "right_predicates": [{"column": "active", "op": "ge", "value": 0}],
    "buckets": 16,
    "sample_size": 500,
    "sample_seed": 424242,
    "parameter_version": 1,
    "boundary_rules": {
      "int": "left_closed_right_open",
      "float": "left_open_right_closed"
    },
    "top_k": 8
  }
}
```

字段约束：

- 必须且只能有两张表；表名唯一；声明行数为正整数；表种子与抽样种子为整数。
- 列名在表内唯一；类型支持 `int`/`integer`/`int64` 与 `float`/`double`/`float64`。
- `domain` 可选，控制生成值域；`unique` 可选，要求值域不小于行数。
- 浮点列可选 `step`，将浮点值量化为离散步长；示例的 `step: 1.0` 让浮点键保持真实可联接的低基数。
- `null_fraction` 可选，范围 `[0, 1)`；整数空值使用 `INT64_MIN` 哨兵，浮点空值使用 `NaN`，联接时排除。
- 谓词运算符：`eq`、`ne`、`lt`、`le`、`gt`、`ge`、`between`、`in`、`not_in`、`is_null`、`not_null`。
- 除 `is_null`/`not_null` 外，空值按 SQL 三值逻辑处理为不命中。
- `parameter_version` 为必填正整数；缺失即 `E_VERSION`。
- `buckets` 必须为正整数；为 `0` 时明确返回 `E_BUCKET`。
- 默认边界规则：整数 `left_closed_right_open`，浮点 `left_open_right_closed`。
- 校验错误会尽可能聚合后整体以 `E_SPEC` 拒绝。

## 确定性生成与摘要

### 数据生成

- 每列使用 `SeedSequence` 派生的 PCG64；派生输入包含表种子、表名、列名和倾斜标记。
- 非倾斜整数列在 `1..domain` 均匀生成；非倾斜浮点列均匀生成后默认保留 3 位小数。
- 倾斜列将 `hot_fraction` 比例的行放入 `domain * hot_domain_fraction` 的热区，其余行在完整值域均匀生成，再执行确定性洗牌。
- `unique` 整数列使用确定性排列，保证无重复。
- 空值位置由同一 PCG64 确定。

### NPZ 与摘要

- NPZ 使用固定 ZIP 时间戳、固定权限、无压缩条目和排序后的列名写出，不依赖 ZIP 当前时间。
- 相同表声明重复生成会得到逐字节相同的 NPZ 文件。
- 每个 NPZ 旁保存 `*.manifest.json`，记录文件 SHA-256、逻辑内容 SHA-256、规范摘要、列类型和生成规则。
- 文件摘要用于检测截断或篡改；逻辑内容摘要先按所有列做规范行排序，因此与行的物理顺序无关。
- 已有 NPZ 与清单摘要不一致时返回 `E_SAMPLE`，不会静默覆盖损坏文件。

## 直方图口径

设过滤后非空键排序数组为 `s`、行数为 `n`、桶数为 `B`，分位边界为：

```text
e_i = s[floor(i * n / B)], i = 0..B
```

报告中的 `quantile_scope` 固定声明为 `full_filtered`，并用 `quantile_sample_size = n` 与 `quantile_basis` 明确说明：分位边界来自过滤后全量键，不使用随机分位草图。

边界归属：

- `left_closed_right_open`：中间桶为 `[e_i, e_{i+1})`；最后一个桶包含右端点，即 `[e_{B-1}, e_B]`。
- `left_open_right_closed`：第一个桶包含左端点，即 `[e_0, e_1]`；后续桶为 `(e_i, e_{i+1}]`。
- 边界归属通过 `searchsorted` 和显式包含标志统一实现；点桶、重复分位边界也遵循同一规则。
- 每个桶记录 `lower`、`upper`、左右包含标志、行数、空值数、区间字符串和排序后键值摘要。
- 空值不参与分位边界和联接，单独记录为 `null_bucket`。

## 双路径估算

### 直方图路径

每个重叠桶对按桶内均匀密度估算：

```text
pair_est = (left_rows / left_width) * (right_rows / right_width) * overlap_width
```

其中整数键宽度按整数点计数，浮点键宽度按数值区间计数，零宽点桶用单点质量处理。所有桶对贡献先量化到 `1e-6` 定点整数，求和后封顶到 `N_left * N_right`，因此不会出现负值或超过两表过滤后行数乘积。

报告输出：

- 重叠桶对数；
- 左右重叠行数比例及组合比例；
- 每个桶对的上下界、宽度、重叠宽度、贡献；
- 整数定点估算值、量化误差上界；
- 与声明规模兼容性检查。

### 抽样路径

样本由“规范行排序 + 数据摘要/谓词摘要/种子派生 PCG64 排列”取得，因此不依赖数组的物理行顺序。

设左右过滤后行数为 `N_L, N_R`，样本规模为 `n`，样本键频次为 `f_L(k), f_R(k)`：

```text
sample_hit_pairs = Σ_k f_L(k) * f_R(k)
est = sample_hit_pairs * (N_L / n) * (N_R / n)
```

报告输出抽样命中对数、抽样键对空间、抽样命中率、左右重标定系数、共同样本键数、逐键频次与贡献。账本中每个键记录样本频次、重标定系数和重标定行频次。

`sample_size` 大于任一表过滤后可用行数时返回 `E_SAMPLE`，并报告 `upper_limit`。

### 路径选择

路径按固定规则选择，不读取真值，保证快照重放一致：

1. 样本命中过少或共同键少于 2：选直方图；
2. 样本规模小于 `2 * buckets`：选直方图；
3. 最大命中键占全部样本命中对至少 50%：选直方图，避免富集键主导抽样方差；
4. 否则比较 `1/(2B)` 与 `1.96/sqrt(n_eff) + 1/sqrt(hit_pairs+1)`，取误差项较小路径。

报告同时给出选择分数、阈值、理由和规则字符串。

## 容差区间与 E_EST

```text
n_eff = min(sample_size, n_left, n_right)
rel_tol = min(0.80, 0.02 + 1.96/sqrt(n_eff) + 1/(2*buckets))
interval = [floor(truth*(1-rel_tol)), ceil(truth*(1+rel_tol))]
```

真值为 0 时使用由两表规模推导的绝对下限。两条路径都必须落在区间内，整体状态才为 `OK`；否则状态为 `E_EST`，报告：

- 每条路径是否区间内；
- 绝对/相对误差；
- 越界比例；
- 选择路径与选择依据；
- 直方图路径、抽样路径各自的主导偏差桶。

CLI 默认仍 `exit 0` 并输出完整诊断；加 `--strict-estimate` 时 `E_EST` 返回退出码 7。

## 分桶级归因

每条估算路径都投影到相同的显式桶对。对每个桶对计算：

```text
difference = histogram_contribution - sampling_contribution
```

排序规则固定为：

1. `absolute_difference` 降序；
2. 左桶下界升序；
3. 左桶下标、右桶下界、右桶下标升序。

分类按稳定优先级：

1. `sample_sparsity`：样本命中对低于稀疏阈值，或任一侧样本桶行为 0；
2. `boundary_membership`：命中点桶、重复等分位边界，或样本键恰好落在显式桶边界；
3. `skew_key_enrichment`：桶对内首键命中占比超过阈值，或倾斜表上达到较低阈值；
4. 残差抽样波动归入 `sample_sparsity`。

每个结果都带 `basis`、`sample_hit_pairs`、边界命中数、首键占比、左右样本桶行数、真值贡献和两项定点贡献。排序、分类和摘要均与输入遍历顺序及数组物理顺序无关。

## 参数演进与陈旧隔离

修改以下参数会令参数版本号在当前会话上严格递增：

- `buckets`
- `sample_size`
- `sample_seed`
- 整数/浮点边界规则

`top_k` 变化也会递增版本，因为它改变当前报告状态。

演进过程：

1. 用新参数完整重算直方图、样本账本、双路径估算、真值、归因、判定；
2. 将旧结果整体标记为 `stale`，不进入当前报告或统计；
3. 当前视图原子替换，不做增量合并；
4. 稳定列出参数差异、估算变化、选择路径变化和受影响重叠桶对。

CLI 示例：

```powershell
py -3 run_experiment.py run --spec examples\demo_skewed.json --workdir var\evolve `
  --evolve-buckets 32 --evolve-sample-size 480 --evolve-sample-seed 778899
```

## 快照与重放

快照为本地 JSON，持久化：

- 规范化场景声明、场景摘要与参数版本；
- 样本逻辑摘要、文件摘要、表行数；
- 直方图完整桶、样本账本摘要；
- 双路径估算、真值、归因、判定；
- 分析总摘要与快照内容摘要。

恢复时会：

1. 校验快照格式、参数版本和表名；
2. 从快照场景重新生成/读取本地样本并核对内容摘要；
3. 按快照参数完整重算；
4. 逐桶、逐字段比较，首个不一致以 `E_REPLAY` 报告 `bucket_lower` 与 `field`；
5. 比较样本账本、双路径定点值、真值、归因摘要、判定摘要和分析摘要；
6. 与当前会话版本不一致时，整体替换数据、参数与估算状态，旧状态标记陈旧；
7. 恢复后当前报告仍显示快照版本；若随后继续演进参数，新版本号会取“当前会话已用上限 + 1”，避免版本回退后再次复用旧版本号。

```powershell
py -3 run_experiment.py run --spec examples\demo_skewed.json --workdir var\snap --snapshot-out var\snap\state.json
py -3 run_experiment.py restore --snapshot var\snap\state.json --workdir var\replay --current-spec examples\demo_skewed.json --report reports\restore.json
```

## 异常注入与结构化日志

内置异常码：

| 代码 | 触发条件 | 退出码 |
|---|---|---:|
| `E_SPEC` | 非法声明、谓词/列/类型错误，聚合拒绝 | 2 |
| `E_BUCKET` | 分桶数为 0 或非正 | 3 |
| `E_VERSION` | 参数版本缺失或快照版本非法 | 4 |
| `E_SAMPLE` | 抽样超上限、NPZ 损坏或摘要不符 | 5 |
| `E_REPLAY` | 快照重算不可复现 | 6 |
| `E_EST` | 至少一条估算路径越出容差区间 | 7（仅 strict 模式） |

示例：

```powershell
py -3 run_experiment.py run --spec examples\demo_skewed.json --workdir var\bad-version --inject-missing-version
py -3 run_experiment.py run --spec examples\demo_skewed.json --workdir var\bad-bucket --inject-buckets-zero
py -3 run_experiment.py run --spec examples\demo_skewed.json --workdir var\bad-sample --inject-sample-corruption
py -3 scripts\make_demo.py
```

样本损坏注入会实际截断工作目录中的 NPZ，并与清单文件 SHA-256 比较。请为注入使用独立工作目录；后续正常重跑同一个损坏目录时仍会返回 `E_SAMPLE`，删除该工作目录即可恢复。

结构化日志同时写入 SQLite 和 JSONL：

- 表 `events(id, timestamp_utc, level, code, event, details_json)`；
- 默认位置：`<workdir>/logs/events.sqlite3`、`<workdir>/logs/events.jsonl`；
- 查看：`py -3 run_experiment.py logs --log-db var\demo\logs\events.sqlite3 --limit 20`。

## 实际验证结果

执行：

```powershell
py -3 -m unittest discover -s tests -v
py -3 scripts\verify_all.py
```

本次结果：

- 15 项单元/端到端测试全部通过；
- 整数倾斜场景：直方图 `541.999999`，抽样 `548.937600`，真值 `542`，区间 `[466, 618]`，双路径均通过，选择直方图；
- 浮点低基数场景：直方图 `95482.707728`，抽样 `86721.474492`，真值 `87955`，区间 `[76015, 99895]`，双路径均通过；
- 连续高基数浮点场景：真值 `91`，区间 `[78, 104]`，直方图 `90627.578429`、抽样 `67.095918`，状态 `E_EST`，两条路径均报告越界比例，并定位主导偏差桶；
- 同一表声明在两个独立目录生成的 NPZ 逐字节一致；
- 参数版本从 `1` 演进到 `2`，旧版本进入 `stale`，受影响桶对 `44` 个；
- 快照重放状态为 `OK`，重算分析摘要与原始摘要一致；
- `E_SPEC`、`E_VERSION`、`E_BUCKET`、`E_SAMPLE` 注入均返回预期代码；
- SQLite 结构化日志可读取。

完整机器可读记录见 `reports/verification.json`。

## 设计边界

- 直方图路径假设桶内键值密度近似均匀；高度稀疏、连续高基数浮点或桶内强偏斜时可能显著高估或低估。归因会给出边界、稀疏或富集键依据。
- 抽样路径在样本命中键对很少时方差较大，因此路径选择会优先改用直方图。
- 容差公式是显式实验口径，不是数据库优化器的统计保证。
- 全量真值执行与全量分位计算均驻留内存，适合本地实验与中等规模样本；超大数据集需要另加外部排序或分块执行。
- 当前环境未执行 Tkinter 窗口交互；Windows 安装 Tcl/Tk 后可直接运行，CLI 与核心引擎已完整验证。
