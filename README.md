# 模型压缩与部署诊断工作台

这是一个完全本地、确定性的模型压缩与部署诊断工作台。它从固定种子的逐层权重包、校准样本、压缩配置和编辑命令流出发，完成逐组分位标定、定点重量化、饱和截断、校准误差归因、软标签蒸馏校正、内容指纹缓存、点预算并发调度、交互式延迟摘要、阶段快照和部署包清单输出。

仓库不依赖 Docker、数据库或外部数据。配置 YAML 由内置的受限解析器读取，不要求安装 PyYAML；计算依赖仅使用 NumPy。

## 快速开始

```bash
python -m pip install numpy==1.26.4 pytest==8.2.2
python -m quant_workbench generate-samples --out samples
python -m quant_workbench validate
python -m quant_workbench run \
  --config samples/config.yaml \
  --weights samples/weights.jsonl \
  --calibration samples/calibration.jsonl \
  --commands samples/commands.jsonl \
  --snapshot .workbench_snapshot \
  --report reports/diagnostic-report.json \
  --html reports/dashboard.html
pytest
```

`run` 会逐条执行命令，在每条命令上按交互延迟预算生成摘要，再在命令流结束后排空待算队列。最终报告包含逐层误差、失效/复用账本、失败明细、调度计划、快照清单以及增量/全量一致性验证结果。可直接打开 `reports/dashboard.html` 查看自包含诊断看板。

## 输入格式

### 权重包 JSONL

每行一个层。层名必须唯一，`shape` 为正整数数组，权重元素必须有限，文件内版本号单调不减。

```json
{"layer":"layer.0","version":1,"shape":[6,8],"weights":[[0.1,0.2]]}
```

### 校准集 JSONL

每个样本含稳定的 `sample_id` 和按层名索引的输入激活：

```json
{"sample_id":"cal-000","activations":{"layer.0":[0.1,0.2],"layer.1":[0.3]}}
```

### 配置 YAML

必要字段包括允许位宽、默认/逐层量化参数、交互延迟预算、线程池大小与上限、单批量化点数预算、快照提交间隔和蒸馏温度。支持两种等价的逐层写法：`layers.<name>` 映射，或 `target_bits`/`group_sizes`/`fractional_bits`/`schemes` 的层名映射。

```yaml
snapshot_version: 1
allowed_bit_widths: [2, 3, 4, 5, 6, 8, 16]
latency_budget_ms: 20
max_threads: 3
thread_limit: 8
point_batch_budget: 64
snapshot_commit_interval: 2
distillation_temperature: 1.25
significant_digits: 7
defaults:
  bits: 8
  group_size: 4
  fractional_bits: 4
  scheme: symmetric
layers:
  layer.0:
    bits: 8
    group_size: 4
    fractional_bits: 5
    scheme: symmetric
```

### 编辑命令流 JSONL

支持 `set_bits`、`set_group_size`、`set_fractional_bits`、`set_scheme`、`set_temperature`、`replace_weight`、`replace_calibration`。`replace_calibration` 可使用命令文件相对路径或内联 `samples` 数组。未知层、形状不符和非有限权重分别产生 `E_LAYER_INVALID`、`E_SHAPE_MISMATCH`、`E_WEIGHT_NOT_FINITE`，并携带命令序号与行号。

```json
{"cmd":"set_bits","layer":"layer.1","bits":4}
{"cmd":"set_temperature","temperature":0.85}
{"cmd":"replace_weight","layer":"layer.4","version":6,"shape":[4,3],"weights":[[0.1]]}
```

## 核心语义

### 标定与重量化

每层按行主序展平并切成 `group_size` 组。每组同时计算：

- 对称缩放：`max(abs(weight)) / ((2^(bits-1))-1)`，零点为 0。
- 非对称缩放：`(max-min) / (2^bits-1)`，零点为 `round(-min/scale)` 并截到码值范围。
- 缩放因子用 float64 计算后按 `significant_digits` 做有效位数裁剪。

所选方案的浮点码值使用 half-away-from-zero 舍入，再按目标位宽饱和。每个元素都记录截断方向、截断前整数、截断后整数和所属分组。小数位宽通过固定点整数 `round(dequantized * 2^fractional_bits)` 实现；小数位宽变化会真实改变 `fixed_integers` 和最终反量化值。`integers`、`fixed_integers`、`dequantized` 与误差均进入稳定输出，因此位宽或方案切换不能只改摘要。

### 校准误差与软标签校正

对每个校准样本，计算 `W·x` 与 `W_dequant·x` 的逐元素偏差。以 `softmax(original_output / temperature)` 作为软标签权重，校正量为 `-(quantized-original) * soft_weight`。每个校正加数按输出索引升序累加，并将 `accumulation_order`、每个加数和运行和写入产物，保证累加顺序可审计。

### 阶段缓存与失效

各阶段使用内容派生 SHA-256 缓存键：

- 标定：层内容指纹 + `bits` + `group_size` + `significant_digits`。
- 重量化：标定键 + `fractional_bits` + `scheme`。
- 校准偏差：重量化键 + 校准集指纹 + `temperature`。
- 部署清单：重量化键 + 校准偏差键 + 层内容指纹。

修改某层位宽、分组或替换单层权重时，该层及其文件顺序上的下游层失效，其他层直接复用。只改蒸馏温度或校准集时，标定和重量化产物复用，仅校准偏差与部署清单失效。只改小数位宽或量化方案时，标定产物直接复用。报告中的 `invalidation_reuse_ledger` 按层名排序，明确记录每条命令、每层、每阶段的 `recompute`/`reuse` 动作、引用配置字段和校准样本集合。

### 交互延迟与陈旧结果

`process_command` 在配置的毫秒预算内最多完成一个点预算批次。若没有排空当前代次，摘要返回最近一次完整结果的指标，同时带 `E_RESULT_STALE`，明确区分：

- `requested_generation`：本次命令期望的代次。
- `result_generation`：摘要指标真实对应的最后完成代次。
- `is_latest`：只有当前代次全部完成时才为 true。
- `current_completed_layers` 与 `progress`：当前代次已完成进度。

交互计算保留少量响应组装余量，并用协作式截止时间检查取消尚未进入下一分组的任务。测试中的模拟层延迟只用于放大 wall-clock 行为，不改变任何计算结果。

### 点数预算与线程池

调度器按层名字典序建立稳定批次。单批点数不超过 `point_batch_budget`，每批最多使用 `max_threads` 个线程；任何批次的完成顺序都不影响最终层顺序。单层点数超过整批预算时，该层及后续队列保持待算并输出 `E_POINT_BUDGET_PENDING`。稳定结果中的 `scheduling.ledger` 与执行线程数无关；测试比较 1 线程和 3 线程的逐层产物、部署清单、调度扣减序列及失败顺序。

### 快照

快照在临时目录中完整构建，然后用目录级替换提交。`manifest.json` 记录全局版本、修订号、配置哈希、校准集哈希、完成层列表、代次和五个阶段的产物版本。`runtime_state.json` 保存命令执行后的有效配置、替换权重和替换校准集，使重启后无需重放已执行命令。

启动时若全局版本或配置哈希不一致，立刻返回 `E_SNAPSHOT_VERSION_MISMATCH`，不逐文件修补、不覆盖任何快照文件。配置未变但校准集变化时，保留可复用的重量化缓存，仅使校准评估和部署清单重新计算。

## 代码结构

- `quant_workbench/loading.py`：YAML/JSONL 加载、schema 校验、命令语义辅助校验。
- `quant_workbench/quantization.py`：分组标定、对称/非对称缩放、定点码值、饱和截断。
- `quant_workbench/calibration.py`：样本前向、软标签校正、逐样本误差、部署条目。
- `quant_workbench/scheduler.py`：稳定点预算计划、线程池批次、协作取消。
- `quant_workbench/engine.py`：增量状态、阶段缓存、命令、延迟摘要、全量重建和快照运行态。
- `quant_workbench/snapshot.py`：版本检查、临时目录构建、整体替换和产物加载。
- `quant_workbench/report.py`：JSON 报告与无外部依赖 HTML 看板。
- `quant_workbench/samples.py`：固定种子样本生成器。
- `tests/`：加载校验、量化、增量一致性、并发、延迟、快照和 CLI/report 测试。

## 已执行验证

本机环境：Linux，Python `3.11.0rc1`，NumPy `1.26.4`，pytest `8.2.2`。最终完整验收结果见 `reports/validation-report.md`、`reports/diagnostic-report.json` 和 `reports/dashboard.html`。当前固定样本结果：

- 5 个层全部完成，部署清单包数 5，校准样本 9 个。
- 6 条命令均在 20 ms 预算内返回；4 条因单批点数预算返回明确陈旧摘要。
- 增量稳定结果与清空缓存全量重建逐字段一致。
- 快照重启结果与全量重建一致，启动恢复未重新计算任何层。
- `pytest -q`：`28 passed`。

实际墙钟数值随机器负载变化；一致性、顺序、代次和版本判定均为确定性检查。

---

## 原始任务文本

从一个空仓库出发，构建一个面向模型压缩与部署的领域可视化与诊断工作台，业务目标是在交互式调整压缩方案与部署资产版本的同时，验证低延迟交互下的计算结果一致性。输入为本地确定性样本：一份按固定种子生成的逐层权重包 JSONL、一份校准样本集、一份压缩与部署配置 YAML（逐层目标位宽、分组粒度、定点小数位宽、量化档位、交互延迟预算毫秒、线程池并发数、快照提交间隔命令数），以及一份编辑命令流。工作台需完成逐层分位标定、定点重量化、饱和截断、反量化误差归因与部署包清单生成，并维护可恢复的阶段快照。难点在于交互延迟预算要求优先返回已完成的可用结果，而定点位宽与量化档位切换又要求内部整数与误差随参数真实变化；同时任意命令序列后的增量运行必须与从零全量重建逐字段一致，且快照整体替换必须识别版本变化。

任务环境与准备：
安装 Python 3.11 或 3.12（Windows 官方安装包），确保 python 与 pip 在 PATH 中。
在空仓库根目录执行 python -m pip install numpy==1.26.4 pytest==8.2.2 安装依赖，版本以本地 wheel 可获取为准。
运行前用固定种子生成逐层权重包 JSONL、校准样本集、压缩配置 YAML 与编辑命令流样本，存放在 samples/ 目录，样本不依赖任何外部下载。
使用 pytest 执行本地测试；交互延迟预算与并发上限在配置 YAML 中给出，测量基于本机墙钟时间，不设定跨平台绝对性能指标。

业务要求：
1. 实现权重包、校准集、配置与命令流的加载校验层：逐行解析权重包 JSONL 并校验层名唯一、形状维度为正整数、权重元素有限、层版本号单调；校验配置中的目标位宽在允许枚举内、分组粒度为正、定点小数位宽与交互延迟预算为正、并发数不超过上限；对未知层、形状不符、非有限权重分别输出 E_LAYER_INVALID、E_SHAPE_MISMATCH、E_WEIGHT_NOT_FINITE 并记录命令序号与行号。
2. 实现逐层定点标定与重量化核心：按配置的目标位宽与分组粒度对每层计算逐组对称与非对称缩放因子与零点，缩放因子用 float64 计算后按有效位数裁剪；把浮点权重转为定点整数并按饱和截断策略处理越界元素，逐元素记录截断方向与截断前定点值，切换位宽或小数位宽必须真实改变内部整数与反量化结果。
3. 实现校准集误差评估与轻量蒸馏校正：在固定校准样本上计算原始权重输出与量化反量化输出的逐样本偏差，按配置的蒸馏温度构造软标签并按软标签权重把量化输出向原始输出校正，输出校正前后逐样本偏差与校正量；校正量计算须显式记录累加顺序，某层被替换后仅该层及其下游产物的偏差失效。
4. 实现基于内容指纹的阶段级精确失效判定：为分位标定、定点重量化、校准偏差与部署包清单各阶段产物计算由层内容指纹、参数指纹与校准集指纹派生的缓存键，并记录每层引用的配置字段与校准样本集合；修改某层位宽、分组粒度或替换单层权重时，仅使该层及其下游失效并纳入重算集合，未失效层必须直接复用且不得重新计算。
5. 实现交互延迟预算内的响应语义：每条命令触达后在配置的交互延迟预算毫秒内优先返回最近一次已完成的最新摘要，含已完成层数、当前最大绝对误差与均方误差、受影响层清单与进度；预算内无法得到最新参数下结果时返回带 E_RESULT_STALE 的摘要并明确标注结果对应代次，不得把旧结果标记为最新。
6. 保证任意命令序列后增量运行与从空白全量重建逐字段一致：支持修改层位宽、切换量化方案、调整分组粒度、修改蒸馏温度、替换单层权重与替换校准样本六类命令，逐层缩放因子、零点、定点整数、反量化值、误差统计、失效复用清单顺序与失败清单顺序都必须与按相同最终权重与配置一次性全量运行完全相同。
7. 实现单批量化点数预算约束下的并发调度与串行一致性：使用线程池并发执行层标定与校准评估，最大并发数取自配置且不得超过；调度器按层名字典序稳定顺序消耗单批量化点数预算，预算不足时保留待算队列并以 E_POINT_BUDGET_PENDING 标记未完成项；每个层任务携带稳定排序键，无论调度顺序与并发度如何，最终层输出顺序、预算扣减序列与失败清单顺序须与串行一致。
8. 实现阶段快照整体替换、版本识别与诊断报告输出：将加载校验、标定、重量化、校准评估、部署包汇总划分为阶段，各阶段产物先写入临时目录再整体替换提交，清单记录全局版本号、配置哈希、校准集哈希、已完成层列表与各产物版本；启动时以版本号与配置哈希判定是否续跑，不一致输出 E_SNAPSHOT_VERSION_MISMATCH 并不得逐文件覆盖；每次运行输出 JSON 诊断报告，含逐层误差、失效复用账本与失败明细。

执行约束：不使用 Docker 或其他容器环境；使用本地确定性样本验证，不访问未授权目标。按上述任务意图交付相应代码、测试或技术说明，记录实际验证结果。

验收：
用本地样本验证逐层压缩结果、预算内摘要语义、增量与全量一致及快照恢复判定，并附报告。
