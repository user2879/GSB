# 本地确定性多路召回与定点融合诊断工作台

本项目从本地 JSONL/JSON 样本出发，执行词面、协同过滤、热度三路召回，随后在两个定点小数位宽下分别完成归一化、通道加权融合、排序特征叠加、裁剪和最终排序。工作台输出逐级误差传播、跨位宽排序翻转归因、边界敏感候选、真值评估、失败清单和可续跑阶段快照。运行期只读取本地文件，依赖 Python 3.10+、标准库和 NumPy，不使用 GPU、容器或网络服务。

## 快速开始

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# POSIX:   source .venv/bin/activate
python -m pip install -r requirements.txt

# 1. 生成固定种子样本和真值
python run.py generate --output data --seed 20261120

# 2. 校验输入；非法行会逐条记录 E_RECORD_INVALID，不会中断其余行
python run.py validate --input data

# 3. 并发执行完整实验
python run.py run --input data --run-dir runs/verification --version 1 --report-dir reports/verification

# 4. 串行重跑以核对逐字段一致性
python run.py run --input data --run-dir runs/verification-serial --version 1 --serial --report-dir reports/verification-serial

# 5. 自动化测试
python -m unittest discover -s tests -v
```

Windows 下将上面的 `python` 保持为 PATH 中的 Python 3.10+ 入口即可。项目不要求安装为包；根目录 `run.py` 会自动把 `src/` 加入导入路径。也可以执行 `python -m pip install -e .` 后使用 `python -m recall_workbench`。

## 输入

默认样本目录结构：

```text
data/
  interactions.jsonl   # user_id, item_id, event_ms, event_type
  items.jsonl          # item_id, terms
  config.json
  truth.json
  sample_manifest.json # generate 命令生成
```

交互记录示例：

```json
{"user_id":"u0001","item_id":"i014","event_ms":1700456789012,"event_type":"click"}
```

物品词典示例：

```json
{"item_id":"i014","terms":["camera","pro","lens","portable"]}
```

`event_ms` 接受非负整数、数字字符串或带/不带时区的 ISO-8601 时间；无时区的 ISO-8601 按 UTC 解释。`event_type` 必须属于配置中的 `event_types`，`item_id` 必须存在于有效词典中。

真值文件由 `generate` 使用固定样本和 24 位参考链路独立计算后落盘：

```json
{
  "schema_version": 1,
  "source_config_hash": "...",
  "oracle": {"fractional_bits": 24, "rounding": "half_even"},
  "top_k": 8,
  "users": {
    "u0001": {"candidates": ["i014", "..."], "ranking": ["i014", "..."]}
  }
}
```

## 配置

完整示例见生成后的 `data/config.json`。关键字段：

- `channels.{lexical,collaborative,popularity}`：`enabled`、`weight`、`candidate_limit`。
- `fusion.strategy`：当前实现 `weighted_sum`。
- `fusion.normalization`：`max` 或 `min_max`。
- `fusion.temperature`：正数；固定点下作为每个通道贡献的除数。
- `fusion.zero_channel_policy`：`redistribute`（默认）或 `none`。
- `ranking_features[]`：支持 `lexical_norm`、`collaborative_norm`、`popularity_norm`、`history_overlap`、`freshness`、`popularity_prior`；顺序即特征贡献分解顺序。
- `fixed_point.primary_bits` / `reference_bits`：主位宽和至少高一档的参考位宽，均为小数位数。
- `fixed_point.tolerance` / `rounding`：误差阈值与确定性舍入方式，当前支持 `half_even`。
- `score_clip.min/max`：最终定点得分的裁剪区间。
- `execution.max_workers` / `snapshot_commit_interval` / `top_k`：并发上限、每阶段快照提交间隔、评估 Top-K。

配置校验是“收集全部错误后拒绝启动”：每项写入 `E_CONFIG_INVALID` 失败记录；JSONL 单条记录错误写入 `E_RECORD_INVALID` 后继续处理其余记录。失败记录按来源、行号、字段、消息、业务码稳定排序。

## 计算语义

### 三路召回

已交互物品从三路候选中排除。每通道先按分数降序、同分按物品 ID 升序截断到 `candidate_limit`，再写入从 1 开始的来源名次。

- 词面：用户历史词项集合与物品词项集合的重合度 `overlap / sqrt(|user_terms| * |item_terms|)`。
- 协同过滤：用户物品共现余弦相似度 `cooc(i,j) / sqrt(count_i * count_j)`，按目标用户历史物品聚合到候选物品。
- 热度：`sum(event_weight[type] * 2^(-(Tmax - event_ms) / half_life_ms))`。

### 归一化与定点融合

对每个通道候选用 `max` 或 `min_max` 归一化，再按 `Q = 2^bits` 转为整数：`q = round_half_even(value * Q)`。后续所有通道权重、归一化值、特征值、特征权重和贡献均使用整数定点运算。

固定顺序：

1. 空通道权重先清零；`redistribute` 时把其定点权重按活跃通道基础权重比例分给活跃通道，整数余数由规范通道顺序中的最后一个活跃通道承接。
2. 每通道贡献 `round_half_even(normalized_q * effective_weight_q / Q)`。
3. 每通道贡献除以定点温度，即 `round_half_even(contribution_q * Q / temperature_q)`。
4. 按 `lexical → collaborative → popularity` 顺序求和得到融合分。
5. 按配置顺序逐项叠加特征贡献 `round_half_even(value_q * weight_q / Q)`。
6. 在定点域裁剪到 `score_clip`。

最终排序键为：`最终定点分降序`、`归一化前名次向量字典序`、`物品 ID 字典序`。名次向量始终使用规范通道顺序；候选在某通道缺席时使用 `candidate_limit + 1` 作为该维名次。

每条候选保留最终得分、融合分、特征分、全部通道来源名次、通道贡献、特征贡献、裁剪标记和定点元数据。

### 双位宽误差归因

同一份精度无关召回证据会分别喂给主位宽和参考位宽，完整重算整条链路。对每个候选比较四个环节：

1. `normalization`：各通道归一化定点值差异的最大值。
2. `channel_weighted_sum`：通道加权求和后的融合分差异。
3. `feature_add`：特征叠加后的裁剪前得分差异。
4. `clipping`：裁剪后的最终得分差异。

差值统一换算到参考位宽的分值单位。首次严格大于 `tolerance` 的环节写入 `first_exceeds_tolerance`。若候选最终位置变化：

- 加权融合阶段的排序位置已变化，归因为 `normalization_residual`。
- 加权融合位置未变、仅最终位置变化，归因为 `feature_rounding_accumulation`。

同时输出相邻分差不超过容差或发生排序变化的边界敏感候选。

### 真值评估

主位宽和参考位宽分别与各自实际最终列表比较，不使用显示层替换。指标包括：

- `hit_rate`：最终 Top-K 命中真值候选数 / 真值候选数。
- `precision_at_k`：命中数 / 最终 Top-K 长度。
- `mean_position_deviation`：共同候选在两份排序中的平均绝对位置差。
- `top_k_overlap`：两套 Top-K 的交集数量。

真值候选未进入最终 Top-K 时，输出其通道来源名次、通道贡献、最终名次和是否曾被召回，便于归因。

## 并发与确定性

- 每个阶段使用 `ThreadPoolExecutor`，并发窗口不超过 `execution.max_workers`。
- 召回任务携带 `user_id|channels=...` 与逐通道 `user_id|channel` 稳定键；融合任务携带 `user_id|fusion=primary,reference` 键。
- 计算结果最终按用户 ID、候选排序键、规范通道顺序和特征配置顺序重排，因此线程完成顺序不会改变输出。
- 失败清单、用户顺序、候选顺序、贡献分解顺序和评估记录均有稳定排序规则。
- `--serial` 使用同一套计算函数逐用户执行，用于逐字段等价验收。

## 阶段快照、信号与续跑

阶段固定为 `recall → fusion → evaluation`。每个阶段：

- 在运行目录同文件系统的临时目录中写入完整 JSON 产物。
- `fsync` 后通过 `os.replace` 原子替换 `stages/<stage>.json`。
- 再原子更新 `manifest.json`，记录 `global_version`、`config_hash`、定点位宽、阶段产物版本、已完成用户列表和游标。
- 每 `snapshot_commit_interval` 个新用户提交一次；阶段结束时提交完整产物。

`SIGINT` / `SIGTERM` 会设置停止请求，不中断正在执行的用户任务；当前计算窗口和待提交结果完成落盘后，以整数退出码 `0` 返回。重启同一运行目录时会按已完成用户集合继续。

续跑前同时校验：

- `run_version` / `global_version` 与 `--version` 一致。
- 配置规范化 JSON 的 SHA-256 与快照一致。
- 阶段产物版本与当前实现一致。

不一致会输出 `E_CONFIG_MISMATCH` 并拒绝续跑。若进程在阶段产物 `os.replace` 之后、清单更新之前退出，重启时以较新的自洽阶段产物为准修复清单。

## 产物

- `runs/<name>/manifest.json`：全局版本、配置哈希、阶段版本、游标、状态。
- `runs/<name>/stages/recall.json`：各用户三路候选、原始分、来源名次、特征输入。
- `runs/<name>/stages/fusion.json`：双位宽完整融合结果、通道/特征贡献、误差归因、边界候选。
- `runs/<name>/stages/evaluation.json`：双位宽评估和未命中归因。
- `runs/<name>/failures.json`：输入失败清单。
- `reports/<name>/experiment_report.json`：完整机器可读报告。
- `reports/<name>/experiment_report.md`：人可读报告。
- `reports/<name>/failures.json`：报告侧失败清单。

`result_digest` 只覆盖实质结果和阶段载荷，排除时间戳、路径和执行模式，因此并发、串行、中断续跑完成后的摘要应一致。

## 业务码

- `E_RECORD_INVALID`：JSONL 或真值条目非法，记录后继续。
- `E_CONFIG_INVALID`：配置非法，拒绝启动。
- `E_CONFIG_MISMATCH`：快照版本或配置哈希与请求不一致，拒绝续跑。
- `E_SNAPSHOT_INVALID`：快照结构损坏或内部不一致。
- `E_INPUT_NOT_FOUND`：必需输入文件缺失。

## 项目结构

```text
src/recall_workbench/
  validation.py   输入与配置校验
  recall.py       三路召回
  fusion.py       定点归一化、融合、特征与排序
  diagnostics.py  四环节误差传播与翻转归因
  evaluation.py   真值评估
  runner.py       并发阶段执行、信号停止、续跑
  snapshot.py     临时目录 + os.replace 原子提交
  report.py       JSON/Markdown 报告
  samplegen.py    固定种子样本与高精度真值生成
  cli.py          命令行入口
tests/            自动化验收测试
data/             固定种子样本
runs/             阶段快照示例
reports/          实验报告示例
docs/             实际验证记录
```

实际验证结果见 `docs/VERIFICATION.md`。
