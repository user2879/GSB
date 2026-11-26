# 定点位置编码与增量账本技术说明

## 一键复现

在 WSL Ubuntu 项目根目录执行：

```bash
bash run_demo.sh
```

脚本会生成样本、配置与命令流，运行全量/增量/预算/并发/恢复验证，并执行 `unittest`。

手动步骤：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/generate_fixtures.py
python -m feedback_pe.demo
PYTHONPATH=. python -m unittest discover -s tests -v
python -m feedback_pe.cli ui
```

WSL 交互视图需要系统 Tk 包：

```bash
sudo apt-get update
sudo apt-get install -y python3.11-venv python3-tk
```

当前交付机的 `sudo` 需要密码，因此验证时把 `libtk8.6`、`libtcl8.6` 及 X/Tk 间接库免 root 解包到了 `.sysroot/root`。`run_demo.sh` 会自动设置 `LD_LIBRARY_PATH`、`TCL_LIBRARY` 与 `TK_LIBRARY`；标准 WSL 安装 `python3-tk` 后不需要这些变量。

## 目录与产物

- `feedback_pe/`：配置/样本/命令校验、量化内核、方案算子、打分、归因、缓存、调度、快照和 Tk 视图。
- `samples/attention_samples.jsonl`：固定整数种子生成的 12 条有效样本和 6 条异常样本。
- `configs/demo_config.json`：维度、方案、基频、缩放、块大小、位宽、档位、容差、并发与预算。
- `commands/demo_commands.jsonl`：方案切换/增删、基频/缩放/块大小、位宽/档位、长度、样本替换/删除和预算命令。
- `state/`：提交快照、内容缓存、账本和预算待办。
- `out/`：权威 JSON 报告、一致性比较、预算、恢复和测试日志。

## 校验与业务码

`feedback_pe/samples.py:69` 的 `validate_sample_record` 逐行收集字段错误并跳过坏记录；`feedback_pe/config.py` 校验配置枚举和数值范围。失败项保持 JSONL 行顺序，包含 `sample_id`、`errors[].code` 与 `errors[].field`。

主要业务码：

- 配置：`E_CONFIG_PARSE`、`E_CONFIG_SCHEMA`、`E_DIMENSION`、`E_BASE_FREQ`、`E_SCALE`、`E_BLOCK_SIZE`、`E_LENGTH_ORDER`、`E_ENUM_FRAC_BITS`、`E_ENUM_QUANT_TIER`、`E_TOLERANCE`、`E_WORKERS`、`E_BUDGET`、`E_SCHEME`。
- 样本：`E_SAMPLE_PARSE`、`E_SAMPLE_ID`、`E_SAMPLE_ID_DUPLICATE`、`E_QUERY_POSITION`、`E_KEY_POSITION`、`E_KEY_POSITION_DUPLICATE`、`E_KEY_VECTORS`、`E_FEATURE_VECTOR`、`E_TRAIN_LENGTH`、`E_TARGET_LENGTH`。
- 命令/调度：`E_COMMAND_PARSE`、`E_COMMAND_OP`、`E_COMMAND_FIELD`、`E_POINT_BUDGET_PENDING`。

## 定点与多方案算子

统一接口见 `feedback_pe/schemes.py`：每个方案都提供 `build_float`、`float_encoding` 和 `fixed_encoding`，返回相同结构的查询/键位置向量、角度、整数码、步长与残差。

- 允许小数位宽：`4,8,12,16`。
- 允许量化档位：`Q8,Q16,Q32`，表示有符号整数总位宽和饱和边界。
- 量化：`feedback_pe/quant.py:19`，记录 `step`、`unclamped_int`、`int_value`、`residual`、`clipped`。
- 方案：`sinusoidal`（几何频率）、`linear_ramp`（线性角度增长）、`blockwave`（分块指数增长）。

## 打分、分层与误差

`feedback_pe/pipeline.py:29` 按维度 `0..D-1` 显式累加整数乘积并保存每个中间累加值；最终乘积和通过 ties-to-even 右移舍入到与位置编码相同的小数 scale。稳定 softmax 先减行最大值，再用 float64 `exp` 和升序分母累加。

分层规则为 `abs(query_position-key_position) < train_length` 进入 `inside_train`，否则进入 `outside_extrapolation`。每层输出样本数、键点数、打分均值、权重均值、最大打分偏差和最大权重偏差。

`feedback_pe/analysis.py:94` 用 float64 参考路径比较位置向量、点积打分和归一化权重。参考值为零时自动切换绝对误差分支，并写入 `absolute_zero_reference` 标记。`out/stage_attribution_profiles.json` 用三组容差剖面证明 position、score、weight 都能成为首次超差环节。

相邻 `Q8 -> Q16 -> Q32` 档位的排序翻转由 `feedback_pe/attribution.py:62` 检测。每条翻转给出旧/新排序、位置量化贡献、累加舍入贡献、归一化减最大值贡献、额外输入特征贡献和未解释 float64 残差，并给出 `attributed_source` 与 `dominant_source`。

## 指纹、复用与精确失效

缓存键在 `feedback_pe/engine.py:43` 构造：

- `position`：样本内容指纹 + 方案参数指纹 + 位宽/档位指纹。
- `score`：依赖 `position` 键和固定累加内核版本。
- `stats`：依赖 `score` 键 + 训练/目标长度指纹。
- `error`：依赖位置/打分钟、长度、容差和翻转档位阶梯。

每个任务的稳定排序键是 `sample_id::scheme::stage`，同时写入 SHA-256 `sort_hash`。账本记录 cache key、样本、方案、阶段、失效依据和复用命中。修改基频只使该方案四个阶段失效；修改训练长度只使 `stats`、`error` 失效，`position`、`score` 直接复用。

## 预算与待办

调度器按样本 ID 稳定排序处理失效样本，单点成本为 `dimension * len(key_positions)`。批次预算不足时，当前样本及其后序列进入待办；下一批待办优先。样本配额通过逐 key/dimension 计量，超过配额立即记 `E_POINT_BUDGET_PENDING`，不提交该样本的半成品。已完成样本在批次提交后不回滚。

## 快照与恢复

`feedback_pe/engine.py:216` 的 `Workspace` 把 `config.json`、`load.json`、`position.json`、`score.json`、`stats.json`、`error.json`、`task_index.json`、`scheduler.json` 和 `ledger.json` 写入临时事务目录，生成全局版本号与配置哈希，再原子重命名到 `state/snapshots/snapshot-XXXXXX`，最后原子替换 `state/CURRENT.json`。

启动只读取 `CURRENT.json` 指向且含 `manifest.json` 的已提交快照；未完成的 `tmp-*` 或 `tx-killed-*` 目录被忽略。强制终止后不会依赖半写入阶段文件。

## 一致性边界

`out/equivalence.json` 对业务报告做规范化 JSON 逐字段比较：最终样本集合、配置、任务键、完整阶段载荷、统计、误差、翻转、边界样本和失败清单完全一致。线程池与串行使用同一配置和同一纯函数 worker，也逐字段一致。

冷启动全量运行的执行账本必然全部是 `recompute`，增量运行会有历史相关的 `reuse`/`invalidate` 事件；因此这些历史状态不纳入业务载荷等价比较。交付另外验证每个复用载荷都等于全新全量结果，且增量最终业务报告与全量重建完全相同。

`out/reuse_recompute_equivalence.json` 进一步记录任务清单顺序、失败清单顺序、增量 `reuse + recompute` 并集与最终 manifest 的一致性；空白全量的 132 个任务全部为 recompute，最终增量批次为 88 个 reuse、44 个 recompute。

## Tkinter 诊断视图

```bash
python -m feedback_pe.cli ui
```

视图直接渲染权威报告，提供方案选择、基频/位宽/档位编辑和增量重算按钮；标签页包括内外分层统计、逐级误差、边界样本、失效/复用账本和业务码计数。控制器只调用同一个 `run_batch`/`build_report` 链路，因此 UI 表格是账本结果的投影。
