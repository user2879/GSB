# 本地确定性合成图像多标签分类实验管线

本工作区实现了一套完全离线、可复现的多标签图像分类实验管线。输入是程序化生成的 PNG/JSON 样本、实验 YAML、规模档位 JSON；管线按“加载校验 → 特征抽取 → 打分 → 阈值判定 → 指标评估”执行，并支持增量编辑、并发、内容寻址缓存、阶段快照、异常注入和 JSON 报告。

交付物包括：

- `synthml/`：生成器、配置校验、特征、三种分类器、自适应策略、快照存储、阶段执行器和 CLI。
- `data/samples/`：当前 72 个确定性合成样本及标注。
- `config/experiment.yaml`：主实验配置。
- `config/experiment_serial.yaml`：并发数 1、快照间隔 4 的对照配置。
- `config/scale_plan.json`：24/3、48/5、72/8 三档规模计划。
- `config/injections.yaml`：截断 PNG、非有限特征、未知标签、阈值越界四类注入。
- `state/`：当前最终快照，保留最近 5 个版本。
- `reports/report.json`：当前最终完整报告。
- `reports/injection_report.json`：异常注入报告。
- `reports/verification_summary.json`：实测结果摘要。
- `tests/`：9 个单元与集成验收测试。

## 1. 环境与安装

要求 Python 3.10 或更高。Windows PowerShell 示例：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import numpy, PIL, yaml; print(numpy.__version__, PIL.__version__, yaml.__version__)"
```

Linux/macOS 示例：

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

运行时依赖仅 `NumPy`、`Pillow`、`PyYAML`。`requirements.txt` 使用 pip 可解析的最新稳定版；本次环境实际锁定为：

```text
Python 3.11.0rc1
numpy 2.4.6
Pillow 13.0.0
PyYAML 6.0.3
```

精确版本见 `requirements-lock.txt`。当前验收容器是 Linux；Windows 使用上面等价的 `py`/`.venv\Scripts\python.exe` 命令。依赖安装完成后，样例生成、训练、增量运行、快照恢复和报告导出均不需要网络。

## 2. 生成样本与配置

从工作区根目录执行：

```bash
.venv/bin/python -m synthml generate \
  --out data/samples \
  --config config/experiment.yaml \
  --plan config/scale_plan.json \
  --injections config/injections.yaml \
  --serial-config config/experiment_serial.yaml \
  --samples 72 \
  --seed 20260317 \
  --width 96 \
  --height 96 \
  --noise 4.0 \
  --blur 0.55
```

Windows 将 `.venv/bin/python` 换成 `.\.venv\Scripts\python.exe` 即可。

生成器按固定 seed 绘制 8 类几何图元：`bar/circle/cross/diamond/ellipse/square/star/triangle`。图元可重叠，样本叠加确定性高斯噪声和可配置高斯模糊。每个样本写为：

```json
{
  "image": "sample_000000.png",
  "labels": ["bar", "circle"],
  "boxes": {
    "bar": [12, 20, 39, 31],
    "circle": [48, 15, 76, 44]
  }
}
```

边界框均为像素整数，格式为 `[x1, y1, x2, y2]`，满足 `0 <= x1 < x2 <= width`、`0 <= y1 < y2 <= height`。

## 3. 运行命令

### 全量重建

忽略已有产物，全部重新计算并发布新快照：

```bash
.venv/bin/python -m synthml run \
  --samples data/samples \
  --config config/experiment.yaml \
  --plan config/scale_plan.json \
  --state state \
  --report reports/report.json \
  --rebuild
```

### 增量运行

复用所有仍有效的内容寻址产物；配置、计划、样本或命令行覆盖变化时只重算受影响阶段及下游：

```bash
.venv/bin/python -m synthml run \
  --samples data/samples \
  --config config/experiment.yaml \
  --plan config/scale_plan.json \
  --state state \
  --report reports/report.json \
  --incremental
```

### 从快照恢复

```bash
.venv/bin/python -m synthml run \
  --samples data/samples \
  --config config/experiment.yaml \
  --plan config/scale_plan.json \
  --state state \
  --report reports/resumed.json \
  --resume
```

`--resume` 要求快照 schema、配置哈希、计划哈希、命令行覆盖和注入档组成的 `run_identity` 全部一致。任何一项不匹配都会输出 `E_CONFIG_MISMATCH`、退出码 3，并拒绝续跑。普通 `--incremental` 允许编辑配置或计划；`--rebuild` 忽略旧产物。

### 导出报告

```bash
.venv/bin/python -m synthml report \
  --state state \
  --out reports/report_export.json
```

### 对照配置

```bash
.venv/bin/python -m synthml run \
  --samples data/samples \
  --config config/experiment_serial.yaml \
  --plan config/scale_plan.json \
  --state state_serial \
  --report reports/serial.json \
  --rebuild
```

主配置并发数为 4、快照间隔 8；对照配置并发数为 1、快照间隔 4。两者的确定性分类结果、失败清单顺序、缓存键顺序和命中计数一致。

### 编辑序列与超参数覆盖

```bash
# 阈值调整：仅判定与指标失效
python -m synthml run ... --incremental --threshold 0.55

# 特征方式和分箱切换：特征与下游失效
python -m synthml run ... --incremental --feature-method hog
python -m synthml run ... --incremental --feature-method color --color-bins 16

# 分类器和 kNN 切换：模型、打分与下游失效
python -m synthml run ... --incremental --classifier knn --knn-k 5

# 精度档位切换：特征量化参数及其下游失效
python -m synthml run ... --incremental --precision high
```

可用分类器：`logistic_regression`、`nearest_centroid`、`knn`。可用特征：`color`、`hog`、`concat`。

### 单样本替换

先准备一个同分辨率、带同名 JSON 的源样本，再替换目标：

```bash
python -m synthml replace-sample \
  --samples data/samples \
  --target sample_000005.png \
  --source data/replacement/sample_000005.png
```

下一次增量运行会按内容指纹自动识别该样本：加载产物只重算该样本，特征产物只重算该样本参与的各规模档；训练集指纹变化会使全部逐样本打分、判定和指标重算。未变化的图像、特征和旧分类器路径继续复用。

## 4. 配置与校验

主配置包含：

- `labels`、`label_rates`、固定 `resolution`。
- 特征方式和分箱候选、HOG 单元、`expected_dimensions`。
- 分类器候选和超参数，包括 Logistic Regression 学习率/轮数/L2、kNN 的 `k_candidates`。
- 多标签阈值、精度档位、并发上限、快照间隔。
- `strategy_candidates` 预置策略表。

关键维度公式：

```text
color_dim   = 3 * color_bins
hog_dim     = hog_bins * hog_cells_y * hog_cells_x
concat_dim  = color_dim + hog_dim
```

配置会检查这些维度与 `expected_dimensions` 声明是否一致；不一致直接给出 `E_CONFIG_INVALID`。未知配置键、空标签集合、非单调规模档、非正并发数、并发数超过上限、非法精度档等都会被拒绝。

规模计划示例：

```json
{
  "schema_version": 1,
  "tiers": [
    {"id": "tier_s", "max_samples": 24, "max_labels": 3},
    {"id": "tier_m", "max_samples": 48, "max_labels": 5},
    {"id": "tier_l", "max_samples": 72, "max_labels": 8}
  ]
}
```

`max_samples` 和 `max_labels` 必须随档位非递减。每个档位选取按标签字典序排序的前 `max_labels` 个标签，以及按样本名排序的前 `max_samples` 个有效样本。

## 5. 算法细节

### 确定性特征

PNG 用 Pillow 解码并转换为连续 RGB `uint8` 数组。图像内容指纹是 RGB 字节的 SHA-256。

- 颜色直方图：每个 RGB 通道按 `bins` 量化、统计、按像素总数归一化，然后拼接。
- HOG：固定整数 Rec.601 灰度，中心差分梯度，0–180° 无符号方向，按固定空间单元累计幅值并做单元内 L1 归一化。
- 拼接：颜色直方图后接 HOG。

所有特征值先执行 `numpy.rint(value * quantization_levels) / levels`，再按 `rounding_decimals` 舍入。特征产物记录 `feature_method`、分箱、HOG 单元、维度、量化档位、舍入位数和舍入模式。相同图像、相同参数重复运行逐元素一致；切换方法或分箱会改变缓存键、维度和数值。

### 三种分类器

- Logistic Regression：逐标签 one-vs-rest；训练样本始终按样本名排序，全批量梯度下降，学习率和轮数来自配置；每条标签逐轮记录损失。
- 最近质心：逐标签计算正类均值与负类均值，使用欧氏距离，原始得分是归一化质心间隔，概率经稳定 sigmoid 映射。
- kNN：邻居按 `(欧氏距离, 样本名)` 排序；原始得分是前 `k` 个邻居的正票比例，概率使用 `(正票 + 1) / (k + 2)` 的 Laplace 平滑。

### 阈值与边界样本

概率均限制在 `[0, 1]` 并记录原始得分。`score > threshold` 判为阳性；`score == threshold` 同样判为阳性，同时写入 `boundary_labels` 和 `boundary=true`。输出标签集合和边界标签均按标签名字典序排列，每个标签还记录 `lexicographic_rank`、比较结果和是否入选。

### 自适应策略

策略选择读取当前档位的实际样本数、标签数、最大/最小类样本数、不平衡比和单样本估计代价。策略表来自配置，候选评分为：

```text
score = base_score
      + scale_bonus[bucket]                 # bucket: small < 32, medium < 64, large
      + imbalance_bonus (if ratio > 1.6)
      + label_bonus * label_count/all_labels
      - cost_weight * estimated_cost

estimated_cost = feature_dimension / 24 * classifier_cost
```

最终记录所有候选的评分拆分、估计代价、最终策略、阈值修正和理由。显式 `--strategy/--feature-method/--classifier/--color-bins/--hog-bins/--knn-k/--threshold/--precision` 覆盖会真正进入特征、模型、打分和判定路径，而不只是改报告字段。

当前 72 样本数据的自适应结果为：

| 档位 | 样本/标签 | 特征 | 分类器 | 阈值 | Macro F1 |
|---|---:|---|---|---:|---:|
| tier_s | 24 / 3 | color | nearest_centroid | 0.50 | 0.88777778 |
| tier_m | 48 / 5 | color | knn | 0.49 | 0.93886062 |
| tier_l | 72 / 8 | concat | logistic_regression | 0.51 | 0.98919465 |

## 6. 缓存、失效与快照

### 阶段缓存键

- 加载：文件 SHA-256、标注、标签模式、分辨率、注入标记。
- 特征：规模档 ID、像素 SHA-256、图像形状、完整抽取参数、精度参数、注入标记。
- 分类模型：规模档、训练集指纹（样本名、像素指纹、有效标签、特征键）、分类器及超参数。
- 逐样本打分：模型键、样本特征键、分类器参数、概率精度。
- 判定：打分键、最终阈值、边界策略。
- 指标：判定键集合、标签、真值、排除样本、阈值、指标精度。

所有阶段产物存放在快照的 `stages/load|feature|score|decision|metric/<cache-key>`。报告中的 `cache_audit` 逐项列出阶段、键、样本、分类器、输入引用、复用/重算状态和原因；`reuse` 与 `invalidations` 是两个可直接审计的子清单。

典型编辑的失效范围：

| 编辑 | load | feature | score | decision | metric |
|---|---|---|---|---|---|
| 阈值调整 | 复用 | 复用 | 复用 | 重算 | 重算 |
| 特征方式/分箱/精度 | 复用 | 重算 | 重算 | 重算 | 重算 |
| 分类器/kNN k | 复用 | 复用 | 重算 | 重算 | 重算 |
| 标注标签变化 | 重算 | 复用 | 重算 | 重算 | 重算 |
| 单样本图像替换 | 1 个重算 | 该样本各档重算 | 全部重算 | 全部重算 | 全部重算 |
| 规模档收窄/扩张 | 复用 | 保留样本复用 | 训练集变化则重算 | 重算 | 重算 |

### 快照发布

运行期间所有阶段产物先写入 `state/.staging-*`。每次提交先写完整清单，再复制为 `state/snapshots/vNNNNNN`，最后原子替换 `state/CURRENT`。清单包含：

- 全局版本号和快照版本号。
- 配置哈希、计划哈希和 `run_identity`。
- 规模档和已完成游标。
- 五个阶段的产物版本哈希。
- 全部产物索引、报告哈希和 `in_progress` 状态。

执行过程中按 `snapshot_interval` 发布 `in_progress` 检查点。最终完成后保留最近 5 个快照，避免清单无限增长；当前快照和紧邻检查点仍可用于恢复。

## 7. 并发与可复现性

特征抽取和逐分类器打分使用 `ThreadPoolExecutor`。每个任务由 `(样本名, 分类器名)` 派生稳定排序键；线程只计算纯函数，产物写入和审计记录按排序键回到主线程完成。最终样本、概率、标签、失败清单、缓存键和命中计数与单线程按样本名有序执行一致。

报告中的阶段耗时使用 `perf_counter`，会随机器和并发度变化；比较全量/增量或并发/串行一致性时应比较确定性字段：`tiers`、`failures`、`scope`、`injection_hits` 以及 `cache_audit` 的键和状态，而不是耗时字段。

## 8. 异常注入与业务码

`config/injections.yaml` 默认启用三个样本级故障：

```bash
python -m synthml run ... --rebuild --injections config/injections.yaml
```

阈值越界项 `INJ-THRESHOLD` 默认关闭，可通过显式 ID 激活：

```bash
python -m synthml run ... --incremental \
  --injections config/injections.yaml \
  --injection-id INJ-THRESHOLD
```

业务码：

- `E_CONFIG_INVALID`：配置、计划、覆盖或注入配置非法；退出码 2。
- `E_CONFIG_MISMATCH`：恢复时 schema/配置/计划/运行身份不匹配；退出码 3。
- `E_SAMPLE_INVALID`：加载阶段样本或标注非法。
- `E_FEATURE_INVALID`：特征非有限值或注入非有限值。
- `E_MODEL_TRAIN` / `E_SCORE_INVALID`：分类模型训练或单样本打分失败。
- `E_INTERNAL`：未预期的运行级异常；退出码 4。

样本级故障不会中断其他样本；运行仍正常发布报告和快照，退出码为 0。每个故障记录样本、阶段、业务码、异常类型、消息和规模档（适用时）；注入命中清单记录注入了什么以及命中次数。

## 9. 报告字段

`reports/report.json` 主要包含：

- `tiers`：每档所选策略、候选评分、理由、特征元数据、分类器诊断、逐样本原始得分/概率/预测标签/边界标签/打分引用、每标签指标和 Macro F1。
- `failures`：稳定排序的失败清单。
- `cache_audit`、`reuse`、`invalidations`、`cache_hit_counts`：逐项复用与重算审计。
- `snapshot`：配置/计划哈希、全局版本、五个阶段版本和 `run_identity`。
- `injection_hits`：注入编号与命中证明。
- `timings`：加载、特征、打分、判定、指标和总耗时。
- `diagnosis`：面向取舍诊断的摘要。
- `environment`：Python、NumPy、Pillow、PyYAML 版本。

## 10. 实际验证

运行全部测试：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

本次结果：`9` 个测试全部通过，耗时约 34 秒。覆盖：

- 特征重复运行逐元素一致、方法/分箱/精度切换改变维度与数值。
- 三种分类器确定性、概率范围、损失历史和 kNN 邻居规则。
- 阈值相等时的边界标签和字典序决策。
- 未知配置键、空标签、维度不匹配、非正并发数、非单调规模档。
- 单样本替换、规模档收窄/扩张、阈值/特征/分类器编辑后的增量结果。
- 增量结果与独立空白全量构建逐字段一致。
- 并发 4 与串行 1 的缓存键、状态、命中计数、打分、指标和失败顺序一致。
- 从 `in_progress` 中间快照恢复并与原全量报告一致。
- 截断 PNG、非有限特征、未知标签的隔离，以及 `run_identity`/配置不匹配拒绝续跑。

对当前 72 样本数据，最终产物共 510 个。无变化增量运行和严格恢复均复用：

```text
load=72, feature=144, score=147, decision=144, metric=3
```

异常注入运行仍处理其余样本并产生 5 条隔离失败；`INJ-001/002/003` 均命中。阈值注入返回 `E_CONFIG_INVALID`，配置或运行身份不匹配返回 `E_CONFIG_MISMATCH`。更完整的实测数字见 `reports/verification_summary.json`。

## 11. 项目结构

```text
synthml/
  cli.py           CLI 与子命令
  config.py        严格配置、计划、注入校验
  generate.py      确定性 PNG/JSON、配置、计划生成器
  imaging.py       PNG 解码、颜色/HOG/拼接特征
  classifiers.py   三种分类器、概率归一化、边界判定
  strategy.py      规模自适应策略评分与覆盖
  store.py         内容寻址产物与阶段快照
  pipeline.py      增量执行、并发、失效审计、报告
tests/
  test_core.py
  test_integration.py
config/
data/samples/
reports/
state/
```

整个实验不依赖 Docker、网络服务或外部模型；数据、缓存键、阈值、模型训练顺序和报告序列均由本地配置与内容指纹决定。
