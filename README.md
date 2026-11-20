# 本地确定性 LLM 偏好对齐奖励建模实验台

这是一个从零实现的本地实验台，用于验证偏好对在 v1 定宽交换格式与 v2 JSON 交换格式之间流转时的互操作、版本兼容、增量失效、快照恢复和训练结果可复现性。项目不依赖 Docker，不访问网络或外部目标；核心实现只使用 Python 标准库，`numpy` 与 `pytest` 仅作为开发/验证依赖。

## 快速开始

要求 Python 3.11 或更高版本。

### Windows

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip --version
python -m pip install -e .
python -m pip install pytest numpy
pref-reward-lab generate --out samples\local_sample --pairs 120 --seed 20260101
pref-reward-lab run --data samples\local_sample --state artifacts\sample_state --out artifacts\sample_output
pref-reward-lab verify --work-root artifacts\verification --pairs 120 --seed 20260101
python -m pytest -q
```

离线时把 wheel 放在本地目录，然后使用：

```powershell
python -m pip install --no-index --find-links .\wheels numpy pytest
```

### Linux / WSL

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e . pytest numpy
pref-reward-lab generate --out samples/local_sample --pairs 120 --seed 20260101
pref-reward-lab run --data samples/local_sample --state artifacts/sample_state --out artifacts/sample_output
pref-reward-lab verify --work-root artifacts/verification --pairs 120 --seed 20260101
python -m pytest -q
```

也可以不安装包，直接运行 `PYTHONPATH=src python3 -m pref_reward_lab ...`。

## 已交付内容

- 定宽 v1 与 JSON v2 解析、配置枚举校验、统一中间表示、稳定失败清单。
- v1 千分比整数到 v2 浮点的精确换算；v2 到 v1 的半入取整；完整旁路元数据与恢复链。
- 子词切分、文本归一化、BLAKE2b 固定哈希、桶计数向量、固定顺序打分和边界判定。
- 规模指标驱动、候选评分、强制档位可测试的策略选择器。
- 特征、打分、转换、评估四阶段内容指纹缓存和逐特征桶引用集合。
- 逐对增量失效、编辑序列增量/全量逐字段一致验证。
- 完整代目录快照、manifest 最后原子替换、配置哈希/版本校验、强杀恢复。
- 线程池并发、稳定任务键、逐对异常隔离、四类真实故障注入和 JSON 实验报告。
- 固定种子样本生成器、CLI、pytest 测试、实际验证报告。

## 输入与输出

样本目录默认包含：

```text
preference_pairs.jsonl   # 偏好对原文
config.json              # 特征、打分、规模档、并发与快照配置
exchange_v1.txt          # v1 定宽交换文件
exchange_v2.json         # v2 JSON 交换文件（数组 / records 对象 / JSONL 均支持）
injections.json          # 可选异常注入配置
```

`preference_pairs.jsonl` 每行字段：

```json
{"pair_id":"pair-000001","prompt":"...","chosen":"...","rejected":"...","annotator_id":"ann-01","timestamp":"2026-01-01T00:00:00Z"}
```

v1 每行严格 48 字符，无分隔符：

```text
pair_id(24, 左对齐) + score_milli(12, 右对齐十进制整数) + delta_milli(12, 右对齐十进制整数)
```

示例：`pair-000001            000000000612000000000123`。非法宽度、非整数、非法 ID、重复 ID 或 v2 标签不在配置枚举内均产生 `E_FORMAT_INVALID`，并进入排序后的失败清单。

v2 记录字段：

```json
{"pair_id":"pair-000001","reward_score":0.6124,"reward_delta":0.1226,"reward_label":"preference","precision_label":"float64"}
```

解析器也接受 `id`、`score`、`delta`、`label`、`precision` 别名。`reward_delta` 缺省为 `0.0`；`reward_label` 与 `precision_label` 必须存在于配置枚举中。

运行输出目录包含：

```text
report.json                  # 完整可复现实验报告
latest_report.json           # 同一报告，便于固定路径读取
converted_v2.json            # v1 -> v2 转换结果
converted_v1.txt             # v2 -> v1 转换结果
sidecar_v2_to_v1.json        # 降级旁路元数据
```

仓库内实际样本在 `samples/local_sample/`，注入运行报告在 `artifacts/injection_report.json`，验证报告在 `artifacts/verification_report.json` 与 `artifacts/verification/`。

## 配置要点

完整示例见 `samples/local_sample/config.json`。关键字段：

- `tokenizer.version`：`subword_v1` 或 `subword_v2`。
- `feature_dim`：每个文本字段最多保留的 token 数。
- `hash_buckets`：基础哈希桶数。
- `normalize_text`：是否执行 NFKC、casefold 与空白折叠。
- `scoring_weights`：`prompt`、`chosen`、`rejected`、`length`、`bias`，全部必须为有限数。
- `reward_delta_threshold`：边界阈值。
- `concurrency`：线程池最大并发数。
- `snapshot_commit_interval`：稳定检查点间隔；每轮仍写入新完整代目录，manifest 记录 `checkpoint_due`。
- `reward_labels` / `precision_labels`：两个版本的标签枚举。
- `source_version`：打分采用的交换版本，`v1` 或 `v2`。
- `strategy_tiers`：规模档位表，含 `max_pairs`、`bucket_multiplier`、`threshold_correction`、`precision_tier`。
- `forced_strategy_tier`：测试或编辑序列可强制档位；强制档位同样真实改变桶数和切分精度。

## 核心语义

### 统一记录

每条有效偏好对输出：

- `reward_score` / `reward_delta`：模型按当前权重计算出的被选得分和奖励差值。
- `model_reward_score` / `model_reward_delta`：同上，显式别名。
- `source_reward_score` / `source_reward_delta`：来自所选 v1/v2 交换文件的源奖励值。
- `chosen_score` / `rejected_score`：被选、被拒回答得分。
- `feature_buckets`、`feature_bucket_count`、`reward_label`、`precision_label`、`source_version`、`boundary`。

边界判定为 `reward_delta <= reward_delta_threshold + selected_tier.threshold_correction`，即恰等于阈值也属于边界样本。聚合报告使用 `math.fsum` 和明确排序；打分结果稳定到 12 位小数。

### 特征与打分

1. 按配置决定是否归一化。
2. 按 `tokenizer.version` 提取词 token；`balanced` 增加二元子词，`fine` 再增加三元子词。
3. 对 `segment|token|切分版本|精度档` 做固定 BLAKE2b 哈希，再对有效桶数取模。
4. prompt、chosen、rejected 分别计数并合并为特征桶向量。
5. 每个桶的极性由固定整数公式生成；按 `prompt -> answer -> length -> bias` 固定顺序用 `math.fsum` 聚合。

同输入、同配置下，线程调度与并发度不影响结果。任务提交顺序由 `sha256(pair_id + ":" + stage)` 派生，最终结果再按 `pair_id` 排序。

### 规模策略

选择器先执行 compact pilot 得到偏好对总数、pilot 唯一桶数、边界比例等指标，再对每个档位计算：

- 容量惩罚；
- 桶碰撞压力；
- 精度成本；
- 边界比例偏差惩罚。

决策记录包含指标数值、每个候选的分数分量、所选档位、有效桶数、精度档、阈值修正和选择理由。切换档位会改变实际参与哈希取模的桶数、切分精度或两者。

### 双向转换

- v1 -> v2：`score_milli / 1000.0`，精度标签为 `milli_exact`。
- v2 -> v1：使用 `Decimal` 的 `ROUND_HALF_UP` 还原千分比；逐对保存原始浮点值、残差、命名标签、精度标签、双向内容指纹。
- 同一运行内先由当前 v2 输入生成旁路，再让 v1 -> v2 恢复链复用该旁路，因此一次升再降后可以逐字段恢复原 v2 内容；旁路中残差使用十进制字符串，避免二次浮点误差。

报告分别给出 `v1_to_v2` 与 `v2_to_v1` 的转换损失、恢复计数和逐对行。

### 缓存与失效

阶段缓存键由上游内容指纹、参数指纹、来源版本和实现 schema 派生：

- `features`：原文 + 切分参数 + 有效桶数 + 精度档 + 来源版本。
- `scoring`：特征内容 + 权重 + 阈值 + 档位参数 + 来源版本。
- `conversion`：交换记录 + 方向 + 标签枚举 + 该对自身的旁路条目指纹。
- `metrics`：全部最终记录 + 失败 + 桶引用 + 指标参数。

替换或删除某对只会使该对的特征、打分和下游指标失效；权重或阈值变化会复用特征，只重算打分及下游；档位或来源版本变化会重算受影响的下游链。每个桶的 `pair_ids` 引用集合写入快照和报告。注入任务带有注入指纹，不会错误复用无注入产物。

### 快照与恢复

`state/generations/gen-<version>-<config-prefix>/` 是完整代目录，包含 `features.json`、`scoring.json`、`conversion.json`、`cache.json` 和 `COMPLETE`。数据先写临时目录，完整后目录原子改名，最后原子替换 `state/manifest.json`。未被 manifest 引用的临时目录和旧代不会被恢复。

manifest 记录全局版本、配置哈希、数据指纹、来源版本、已完成轮次游标、检查点间隔和阶段指纹。`--resume` 会验证配置哈希、可选版本号和可选来源版本；不一致返回 `E_CONFIG_MISMATCH` 并拒绝复用。普通编辑运行使用逐阶段内容键复用未失效产物，因此配置变化后的增量编辑仍可继续。

### 并发与注入

特征与打分使用 `ThreadPoolExecutor`。注入配置示例：

```json
{"id":"INJ-001","pair_id":"pair-000001","stage":"features","kind":"zero_hash_buckets"}
```

支持：

| kind | stage | 业务码 | 行为 |
| --- | --- | --- | --- |
| `zero_hash_buckets` | `features` | `E_HASH_BUCKETS_ZERO` | 对该对真实执行零桶提取并失败 |
| `non_finite_weight` | `scoring` | `E_NONFINITE_WEIGHT` | 对该对传入 NaN 权重并失败 |
| `boundary_equal_threshold` | `scoring` | `W_BOUNDARY_EQUAL_THRESHOLD` | 将差值置为有效阈值，命中边界警告 |
| `missing_sidecar` | `conversion` | `E_SIDECAR_MISSING` | 对所选方向真实触发旁路缺失 |

注入按对隔离，其他偏好对继续计算；每个注入项的报告包含 `hit`、阶段、业务码和说明。

## CLI

```text
pref-reward-lab generate --out DIR --pairs 120 --seed 20260101
pref-reward-lab run --data DIR [--config FILE] [--state DIR] [--out DIR]
                    [--source-version v1|v2] [--resume] [--expect-version N]
                    [--force-recompute] [--crash-after-stage features|scoring|conversion|cache]
pref-reward-lab edit --data DIR --edit-file edits.json [--run-each]
pref-reward-lab inspect-snapshot --state DIR [--config FILE] [--expect-version N]
pref-reward-lab verify --work-root DIR --pairs 120 --seed 20260101
```

编辑文件是操作数组，例如：

```json
[
  {"op":"force_strategy_tier","tier":"large"},
  {"op":"switch_version","version":"v1"},
  {"op":"replace_pair","pair_id":"pair-000010","changes":{"prompt":"prism quiet raven","v1_score_milli":777,"v2_score":0.7774}},
  {"op":"delete_pair","pair_id":"pair-000011"},
  {"op":"set_weights","weights":{"prompt":0.2,"chosen":1.1,"rejected":0.9,"length":0.03,"bias":0.0}},
  {"op":"set_threshold","value":0.07}
]
```

## Python API

```python
from pref_reward_lab import ExperimentLab, generate_dataset

generate_dataset("samples/demo", pair_count=120, seed=20260101)
lab = ExperimentLab("samples/demo", state_dir="state", output_dir="output")
report = lab.run()
lab.apply_edit({"op": "set_threshold", "value": 0.07})
incremental = lab.run()
```

底层转换与解析函数可直接使用：`parse_v1_text`、`parse_v2_text`、`convert_v1_to_v2`、`convert_v2_to_v1`。

## 错误码

| 业务码 | 含义 |
| --- | --- |
| `E_FORMAT_INVALID` | 定宽、JSON、字段类型、重复 ID 或标签枚举非法 |
| `E_CONFIG_INVALID` | 配置结构、数值、档位或字段非法 |
| `E_NONFINITE_WEIGHT` | 打分权重非有限 |
| `E_HASH_BUCKETS_ZERO` | 特征提取桶数为零 |
| `W_BOUNDARY_EQUAL_THRESHOLD` | 奖励差值恰等于有效阈值 |
| `E_SIDECAR_MISSING` | 恢复链要求的旁路元数据缺失 |
| `E_CONFIG_MISMATCH` | 恢复请求的配置哈希或版本不一致 |
| `E_SNAPSHOT_INCOMPLETE` | manifest 或代目录不完整 |
| `E_INJECTION_TARGET_NOT_FOUND` | 注入目标偏好对不存在 |
| `E_INTERNAL` | 单对任务中的非预期异常，已隔离 |

## 实际验证结果

在当前 Python 3.11 / WSL 环境执行 `python -m pytest -q`，17 个测试全部通过。实现只使用跨平台标准库路径与原子替换接口；Windows 使用方式见快速开始。

固定种子 120 对样本的注入运行结果：

- 有效打分记录 118 条；2 条分别因零桶与 NaN 权重注入被隔离。
- 边界样本 47 条；四类注入全部 `hit=true`。
- 失败业务码：`E_FORMAT_INVALID`、`E_HASH_BUCKETS_ZERO`、`E_NONFINITE_WEIGHT`、`E_SIDECAR_MISSING`、`W_BOUNDARY_EQUAL_THRESHOLD`。
- 策略选择 `medium`：有效 256 桶、`balanced` 精度、阈值修正 `0.01`。
- v2 -> v1 最大分数残差 `0.40000000000003` 毫分、最大差值残差 `0.300000000000002` 毫分；v1 -> v2 旁路恢复 119 条，损失为 0。
- 快照全局版本 1，代目录 `gen-00000001-5d137ba968a4`。
- 本次验证从空 state/output 开始；交付的 `artifacts/sample_state` 仅包含上述一个完整代，无旧快照或崩溃临时目录。

`pref-reward-lab verify` 使用同一 120 对样本施加 9 次编辑（档位切换、v1/v2 切换、替换、删除、权重与阈值调整），结果：

- 每次编辑后的增量结果与空白状态全量重建：全部逐字段一致。
- 并发 1 与并发 8：核心结果一致。
- 旁路恢复：119 条全部逐字段恢复，零 mismatch。
- 强制终止：只恢复上一完整快照，恢复点为第 1 轮。

可复核：`artifacts/injection_report.json`、`artifacts/verification_report.json`。

## 文档与测试

- `docs/format-spec.md`：格式、精度和旁路字段规范。
- `docs/architecture.md`：模块、阶段缓存、快照和编辑一致性设计。
- `tests/`：17 个 pytest 用例。
- `requirements-dev.txt`：开发依赖。

## 可复现性边界

报告核心指纹排除缓存计数、快照版本和输出目录等运行历史字段；它覆盖策略、统一记录、边界清单、桶汇总与引用、失败清单、双版本转换产物、指标和注入结果。因此同一最终输入与配置在不同历史、不同并发度、不同输出目录下具有相同核心指纹。快照版本与复用计数仍然保留在报告中，用于审计增量执行过程。
