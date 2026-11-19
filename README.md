# JoinLab：等值联接规模估计与采样诊断工作台

JoinLab 是一个完全本地运行的 Windows 桌面工作台，用于比较等值联接的两条规模估计路径，并把差异归因到具体分位桶，最后用 NumPy 真值联接验证估计是否落入显式容差区间。除 NumPy 外只使用 Python 标准库（`sqlite3`、`tkinter`、`unittest` 等），不需要 Docker、数据库服务或网络访问。

## 已交付能力

- JSON 基表定义、确定性行摘要、强校验和原子 SQLite 灌入。
- 整数/浮点联接键分位直方图，边界归属规则显式声明。
- 基于种子与数据摘要的确定性哈希优先级抽样，并保留键频次。
- 直方图重叠路径、抽样命中率路径、百万分之一整数定点结果。
- 桶级贡献差归因，稳定排序并分类为“边界归属 / 样本稀疏 / 倾斜键富集”。
- NumPy 真值等值联接、稳定输出行序、分组贡献和容差判定。
- 参数版本单调递增、陈旧结论隔离、整体替换报告。
- 快照保存、恢复重算、逐项比对与 `E_REPLAY` 不可复现定位。
- Tkinter 图形工作台、CLI、Windows 批处理入口和 17 项回归测试。
- `E_SKEW`、`E_TYPE`、`E_SNAP` 三类异常注入及 JSONL 结构化日志。

## 目录

```text
demo.py                     CLI：run / restore / inject / verify
workbench.py                Tkinter 桌面工作台
joinlab/spec.py             JSON 校验、确定性摘要、SQLite 灌入/回读
joinlab/histogram.py        分位直方图与边界规则
joinlab/sampling.py         确定性抽样与频次
joinlab/estimation.py       双路径估计、定点与桶级归因
joinlab/truth.py            NumPy 真值联接与容差判定
joinlab/session.py          参数版本、整体替换、已装载会话
joinlab/snapshot.py         快照持久化与恢复比对
joinlab/anomaly.py          三类异常输入注入
samples/                    确定性样本、JSON 定义与会话参数
tests/test_joinlab.py       unittest 回归测试
artifacts/                  运行生成的 SQLite、快照、日志和验收报告
```

## 环境准备（Windows）

1. 安装 Python 3.10 或以上，并勾选 Tcl/Tk。确认：

```bat
python --version
python -c "import sqlite3, tkinter; print(sqlite3.sqlite_version, tkinter.TkVersion)"
```

2. 安装 NumPy：

```bat
python -m pip install -r requirements.txt
```

3. 运行时设置 UTF-8。仓库内的 `.bat` 已自动设置，手工运行时执行：

```bat
set PYTHONIOENCODING=utf-8
```

## 快速开始

```bat
REM 完整本地验收：双路径、归因、真值、替换、恢复、三类异常
python demo.py verify

REM 生成一次完整报告
python demo.py run --output artifacts\run_report.json

REM 图形工作台
python workbench.py

REM 回归测试
python -m unittest discover -s tests -v
```

也可以直接双击 `run_verify.bat`、`run_demo.bat`、`run_workbench.bat` 或 `run_tests.bat`。

## 输入格式

工作区 JSON 顶层包含 `workspace_id`、`tables`、`join`、可选 `filters`、可选默认 `params` 和 `attribution_k`。样本见 `samples/base_tables.json`。

```json
{
  "workspace_id": "joinlab-local-demo-v1",
  "attribution_k": 5,
  "tables": [
    {
      "name": "accounts",
      "row_count": 720,
      "skew": false,
      "columns": [
        {"name": "account_id", "type": "integer"},
        {"name": "segment", "type": "integer"}
      ],
      "data_file": "rows/accounts.json"
    }
  ],
  "join": {
    "left_table": "accounts",
    "left_key": "account_id",
    "right_table": "events",
    "right_key": "account_id"
  },
  "filters": {
    "accounts": [{"column": "segment", "op": "eq", "value": 1}],
    "events": [{"column": "amount", "op": "ge", "value": 100.0}]
  }
}
```

行文件可以写 `{"rows": [[...], ...]}`，也可以直接写行数组；行内可用数组或按列名对象。`data_file` 相对于工作区 JSON 所在目录解析。列类型支持 `integer/int/int64` 与 `float/float64/double/real`；联接允许整数与浮点数值互比，其他类型报 `E_TYPE`。

过滤操作符：`eq/ne/lt/le/gt/ge`（含 `= != < <= > >=`）以及 `is_null/not_null`。过滤先于直方图、抽样和真值联接执行；因此抽样规模上限按过滤后的行数检查。

会话参数文件 `samples/session.json`：

```json
{
  "bucket_count": 10,
  "sample_size": 320,
  "seed": 20261120,
  "quantile_policy": "nearest_rank",
  "attribution_k": 5
}
```

## 确定性规则与核心口径

### 1. 装载与摘要

- 表名和列名必须唯一，联接键必须存在，`row_count` 必须与实际行数一致。
- 行按“列类型标签 + 值规范表示”排序，再以表名、行值、重复序号生成稳定 `__rid`。因此源行顺序改变不会改变数据摘要、样本集合或真值输出摘要。
- `structure_version` 只描述表名、列名/类型和声明行数；`data_digest` 描述规范化后的全部行。SQLite 灌入完成后会重新回读并复算摘要，不一致整体回滚。
- SQLite 文件使用临时文件构建后原子替换，不会留下半成品。
- 联接键允许空值；直方图单独记录总空值数，真值联接排除空键。桶内 `null_count` 固定为 0。
- 当表声明 `skew=true` 且联接键最高频值占非空行至少 65% 时，装载以 `E_SKEW` 拒绝；正常数据可以把 `skew` 作为诊断标记保留。

### 2. 分位直方图

- 桶数为 `B`，有效非空行数为 `N`；`B <= 0`、`B > 过滤后行数` 或 `B > N` 报 `E_HIST`。
- `nearest_rank`：整数键边界取排序后第 `floor(b*N/B)` 个值；浮点键取第 `floor(b*N/B)-1` 个值以配合右闭区间。
- `lower_rank`：两种类型都取前一个秩位置，形成不同分位口径；口径变化会进入参数版本。
- 整数键：所有桶为 `[lower, upper)`，末桶 `upper=max+1` 是显式排他哨兵。边界值 `x == upper` 进入下一桶。
- 浮点键：首桶 `[lower, upper]`，后续桶 `(lower, upper]`。边界值归入包含右端点的左侧桶。
- 每桶记录上下界、包含性、行数、空值数、唯一键数、最高频键和频次。相同数据与参数重复构建得到相同规范 JSON、桶边界、行数序列和 SHA-256 摘要。
- 重复值跨越分位切点时可能出现空桶；这是保持值域不重叠和边界归属一致的结果。

### 3. 确定性抽样

- 每个过滤后行的优先级为 `SHA-256(seed + data_digest + __rid)`，按优先级升序取前 `sample_size` 行，再按 `__rid` 排序记录。
- 因此样本与源文件物理顺序无关；相同种子、规模、表数据摘要得到相同样本键集合与样本摘要。
- 样本保留每个非空键的出现次数；空键单独计数。`sample_size > 过滤后行数` 报 `E_SAMPLE`。

### 4. 两条估计路径

定点统一为 `scale = 1_000_000`，截断方向为 `floor_toward_zero`。所有 `*_fixed` 字段是“行数 × scale”向下截断后的整数。

直方图路径对每个重叠桶对：

```text
rL = 重叠区间长度 / 左桶长度
rR = 重叠区间长度 / 右桶长度
cL = 左桶行数 × rL
cR = 右桶行数 × rR
D  = 整数键：重叠区间的整数点数
     浮点键：max(左桶唯一键数, 右桶唯一键数)
桶对贡献 = cL × cR / D
直方图估计 = floor(sum(桶对贡献) × scale)
```

候选行数比例和桶对贡献同时输出；整数域使用区间点数可避免把键域空洞误当成稠密键。浮点域使用唯一键数近似离散键域。

抽样路径：

```text
sample_hits = sum(左样本键频次 × 右样本键频次)
combined_rescale = (左全表行数 / 左样本行数) × (右全表行数 / 右样本行数)
抽样估计 = floor(sample_hits × combined_rescale × scale)
```

### 5. 桶级归因

- 对每个重叠桶对计算 `difference_fixed = histogram_contribution_fixed - sample_contribution_fixed`。
- 最终排序键固定为：`|difference_fixed|` 降序、键区间下界升序、左桶号升序、右桶号升序；输出前 K 个非零差异桶。
- 分类优先级：桶内最高频键占比至少 45% → “倾斜键富集”；任一侧桶样本键行数少于 5 或样本命中少于 2 → “样本稀疏”；否则 → “边界归属”。
- 遍历桶时显式按桶号排序，桶对再统一排序，因此比较顺序不影响差异桶序列和来源分类。

### 6. 真值联接与容差

- 过滤后联接键转成 NumPy 数组，`np.intersect1d` 求公共键，每个键用 `np.repeat/np.tile` 形成笛卡尔匹配，最后 `np.lexsort`。
- 输出行序固定为 `key ASC, left.__rid ASC, right.__rid ASC`，空键排除；分组贡献也按键升序。输出序列和分组分布分别记录摘要。
- 直方图容差基点为 `clamp(20000 / B, 1000, 5000)`；抽样容差基点为 `clamp(20000 / isqrt(sample_size), 1000, 5000)`。
- 容差区间还加入 `max(1, B/2)` 行绝对松弛。估计未落入任一对应区间时报 `E_ESTIMATE`，附带首个差异键区间、两路径估计值、真实行数和两个区间。

## 参数版本、陈旧防护与快照

- 分桶数、抽样规模、种子、分位口径或 Top K 改变时，版本状态文件中的版本号严格递增。
- 修改参数会先冻结旧视图，写入 `stale_epochs`（`stale=true`、旧版本、旧摘要、旧估计与归因摘要），随后清空当前视图并从同一新参数全量重算。
- 当前报告的 `histograms`、`samples`、`estimate`、`attribution`、`validation` 只包含一个参数版本，不进行增量合并。
- 替换报告稳定列出变化表、旧/新桶数、旧/新直方图与样本摘要，以及受影响的四类结论。
- 快照持久化表结构与数据摘要、过滤摘要、参数版本、直方图/样本摘要、估计、归因、真值和判定。恢复时整体替换当前视图，并按快照参数重算。
- 重算完成后逐项比较表摘要、直方图、样本、估计、归因、真值和判定；任何不一致报 `E_REPLAY` 并给出 `first_mismatch.path/expected/actual`。快照缺少 `params_version` 报 `E_SNAP`。
- 恢复为了保真会采用快照的参数版本；之后再次修改参数时，版本生成器仍会从历史状态中的最大值继续递增。

## 异常、日志与注入

错误输出为结构化 JSON，包含 `code`、`message`、可选 `field_path` 和 `details`。所有业务错误同时写入 `artifacts/logs/joinlab.jsonl`。

```bat
python demo.py inject --kind skew
python demo.py inject --kind type
python demo.py inject --kind snapshot_version_missing
```

三类注入只复制/修改输入文件或快照文件，再真实走装载/恢复逻辑：

| 注入 | 预期码 | 典型字段路径 |
|---|---|---|
| 右表联接键集中到单个值并设置 skew | `E_SKEW` | `tables[1].skew` |
| 左联接键类型改为 `text` | `E_TYPE` | `tables[0].columns[0].type` |
| 快照删除 `params_version` | `E_SNAP` | `snapshot.params_version` |

其他业务码：`E_TABLE`（表/列/行/过滤定义、重复表名、行数不一致、联接键缺失）、`E_HIST`（桶数或分位口径）、`E_SAMPLE`（样本规模或种子）、`E_ESTIMATE`（真值容差失败）、`E_REPLAY`（快照不可复现）。

## 图形工作台

`workbench.py` 提供：

- 工作区、会话参数、SQLite、快照和版本状态路径选择。
- 分桶数、抽样规模、种子、分位口径和 Top K 编辑。
- “装载并全量重算”“应用参数并整体替换”“当前参数重算”“保存快照”“恢复快照并逐项校验”。
- 表级摘要树，以及估算、归因、真值、版本/陈旧、完整 JSON 五个页签。
- 错误码、字段路径和详情直接显示，并写入同一结构化日志。

当前 CI/验证容器没有安装 Tkinter，因此本次实测对 GUI 做了全文件语法编译检查；核心引擎、CLI、测试和报告均实际运行。Windows 官方 Python 正常安装 Tcl/Tk 后即可使用 GUI。

## 实际验证结果

验证环境：Linux x86_64、Python 3.11.0rc1、NumPy 2.4.6、SQLite 3.37.2；未使用容器化数据库或其他服务。样本参数为 `B=10, sample_size=320, seed=20261120, nearest_rank`。

| 指标 | 结果 |
|---|---|
| 账户表 | 720 行，过滤后 504 行，空联接键单独计数 |
| 事件表 | 1080 行，过滤后 555 行 |
| 直方图估计 | 386.211082（fixed `386211082`） |
| 抽样估计 | 376.966406（fixed `376966406`） |
| 真实 NumPy 联接 | 386 行 |
| 直方图容差区间 | `[303, 469]`，通过 |
| 抽样容差区间 | `[335, 437]`，通过 |
| 样本命中对 / 合并重标定因子 | `138` / `2731640.625000` |
| 双次构建报告摘要 | 均为 `b8ca136788e770173686514dfca4c10fbe74cd0dbbd678ee3f67b9819c3fc9a1` |
| 参数整体替换 | 版本递增，旧视图标记 STALE，两张表旧/新桶数与摘要均列出 |
| 快照恢复 | 表、直方图、样本、估计、归因、真值、判定、结论摘要逐项 MATCH |
| 异常业务码 | `E_SKEW`、`E_TYPE`、`E_SNAP` 全部命中；篡改快照恢复命中 `E_REPLAY` |
| 回归测试 | `17/17 OK` |

完整机器可读结果在 `artifacts/demo_report.json`，单次完整报告在 `artifacts/run_report.json`，快照在 `artifacts/snapshot.json`，结构化日志在 `artifacts/logs/joinlab.jsonl`。`demo.py verify` 还会生成被篡改快照和四类注入输入，分别位于 `artifacts/injected/` 下。

## 已知边界

- 直方图估计假设桶内键近似均匀；强倾斜会由抽样路径和归因分类揭示，但不应把直方图结果当作精确答案。
- 浮点键把桶内唯一键数作为离散域近似，不尝试推断未观察到的浮点值分布。
- 整数与浮点互比时 NumPy 真值联接统一转为 `float64`；超过 `2^53` 的极大整数与浮点混合联接可能存在精度偏差。
- 当前实现面向单次两表等值联接和本地规模数据；超大数据集需要改为分块真值执行与磁盘快照。
