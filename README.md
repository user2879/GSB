# 多路召回候选分数融合应用

本项目实现一个完全本地、确定性的推荐系统多路召回路分数融合器，重点覆盖：

- 通道内确定性归一化、定点量化与残差记录；
- 按通道权重和固定顺序进行多路分数合并、截断与贡献分解；
- 基于候选级/融合级内容指纹的精确增量失效与直接复用；
- 全局预算、单通道配额和线程池执行的可审计调度；
- 版本化阶段快照、原子提交、崩溃恢复和 `E_SNAPSHOT_STALE` 诊断；
- 四类真实输入故障注入、float64 参考路径和完整 JSON 报告。

整个验证过程不需要 Docker、网络、账号或外部服务。

## 1. 环境与快速运行

当前实际验证环境：

- Python `3.11.0rc1`，满足 Python >= 3.10；
- numpy `1.26.4`，来自本机离线环境；
- Linux/WSL2；
- 当前系统的 `ensurepip` 不可用，因此项目虚拟环境使用
  `python3 -m venv --without-pip --system-site-packages .venv` 创建，不下载任何包。

```bash
# 生成固定种子样本（已生成时不会覆盖）
.venv/bin/python -m candidate_fusion init-samples --root samples

# 跑完整验收矩阵，生成 artifacts/acceptance/summary.json 及全部报告
.venv/bin/python -m candidate_fusion acceptance \
  --samples samples \
  --output artifacts/acceptance \
  --force-fixtures

# 单元测试
.venv/bin/python -m unittest discover -s tests -v

# 单次运行；未指定 --report 时默认写入 <workspace>/report.json
.venv/bin/python -m candidate_fusion run \
  --workspace samples/base_workspace \
  --report artifacts/acceptance/manual_run.json

# 应用单条/多条编辑命令
.venv/bin/python -m candidate_fusion apply-edits \
  --workspace samples/base_workspace \
  --sequence samples/edits/sequence.json

# 模拟写入阶段临时目录后、切换 CURRENT 前被强制终止
.venv/bin/python -m candidate_fusion run \
  --workspace samples/base_workspace \
  --simulate-crash-after normalize
```

`--simulate-crash-after` 正常返回码为 `75`，用于证明下一次启动只能看到上一次完整提交。

## 2. 项目结构

```text
candidate_fusion/
  config.py       配置校验、配置哈希
  loader.py       JSONL 加载、容量/唯一性/有限数校验、故障注入
  normalize.py    通道内 minmax/zscore、Decimal 定点化、残差
  reference.py    独立 numpy float64 参考路径
  fusion.py       贡献分解、固定顺序融合、截断、边界诊断
  scheduler.py    预算预分配、线程池执行、并发观测
  snapshot.py     阶段快照、哈希清单、临时目录与原子 CURRENT
  engine.py       增量/全量主流程与报告
  samples.py      固定种子样本生成
  edits.py        编辑命令应用
  acceptance.py   端到端验收矩阵
tests/
  test_fusion_app.py
samples/
  base_workspace/
  edits/
  injections/
artifacts/acceptance/
  summary.json
  reports/
  budget/
  injections/
  snapshot/
  final_snapshot/
```

## 3. 输入与配置

工作区格式：

```text
workspace/
  config.json
  fragments/
    ch_alpha/frag_000.jsonl
    ch_alpha/frag_001.jsonl
    ch_beta/frag_000.jsonl
  state/                   # 运行后生成
  report.json              # CLI 默认报告
```

每个 JSONL 行包含：

```json
{"candidate_id":"cand_001","channel":"ch_alpha","score":0.25,"features":[0.1,0.2]}
```

配置包含通道容量和权重、特征维度、归一化方式、量化档位、舍入方式、截断上限、线程数、全局预算与单通道配额。完整示例见 `samples/base_workspace/config.json`。

加载层逐行处理，非法条目进入稳定排序的失败清单并继续处理其他条目：

| 输入问题 | 业务码 |
|---|---|
| 配置字段非法、通道名不匹配、重复候选、维度错误、容量超限、非有限特征 | `E_CONFIG_INVALID` |
| 得分非有限 | `E_NONFINITE_SCORE` |

重复候选保留首个合法条目，后续重复项报错；容量按通过校验的候选数计算。

## 4. 确定性归一化与定点化

统计量与逐候选计算顺序固定为 `candidate_id` 字典序升序。

- `minmax`: `(score - min) / (max - min)`；
- `zscore`: `(score - mean) / population_std`；
- 分母为 0 时，该通道全部归一化值强制为 `0.0`，并记录
  `E_NORMALIZE_ZERO_DENOM` 事件，不伪造异常分数。

定点量化：

```text
step       = 2^(-quant_bits)
fixed      = round_half_even/half_up/truncate(normalized / step)
dequant    = fixed * step
residual   = normalized - dequant
```

实现使用 `Decimal(str(normalized))` 进行舍入，避免平台浮点舍入差异。产物记录 `fixed`、`quant_step`、`residual`、`norm` 和指纹。

## 5. 指纹、融合与失效口径

候选指纹以规范化 JSON 的 SHA-256 计算，内容元组为：

```text
schema, channel, candidate_id, fixed, feature_dim
```

融合指纹内容元组为：

```text
schema, candidate_id, 有序候选指纹列表, 有序(channel, weight), trunc_k
```

融合规则：

1. 通道按名称字典序遍历；
2. 同一候选的贡献按通道名升序累加；
3. 每个贡献记录 `order`、`fixed`、`dequantized`、`weight`、`weighted`、候选指纹；
4. 按融合分数降序、候选标识升序截断；
5. `trunc_k=0` 返回空集并记录 `E_TRUNC_ZERO`。

失效矩阵：

| 编辑 | 候选定点状态 | 融合结果 | 预算计算范围 |
|---|---|---|---|
| 追加/替换单通道碎片 | 该通道重新归一化，通道内候选失效；其他通道复用 | 仅涉及该通道候选的融合失效 | 仅失效融合的贡献 |
| 修改通道权重 | 全部候选复用 | 仅含该通道贡献的融合失效 | 仅失效融合的贡献 |
| 修改截断上限 | 全部候选复用 | 融合指纹含 `trunc_k`，全部融合失效 | 全部融合的贡献 |
| 切换归一化/量化/舍入 | 全部候选失效 | 全部融合失效 | 全部融合的贡献 |

未失效融合直接从上一完整快照的 `fuse.json` 读取记录，不进入预算任务。无编辑重跑时实际预算消耗为 0。

## 6. 预算与并行调度

一个预算单位是“一个通道-候选贡献的重算”。所有预算在提交线程任务之前按通道名字典序串行预留，因此审计序列不随线程完成时机漂移。

```text
granted = min(requested, channel_quota, remaining_global)
```

- 超过单通道配额：记录 `E_CHANNEL_QUOTA_EXHAUSTED`，剩余贡献进入 `pending_queue`；
- 全局预算不足：记录 `E_GLOBAL_BUDGET_EXHAUSTED`，后续未完成通道继续记录；
- `audit` 固定记录 `seq/channel/requested/granted/remaining_global_after/status`；
- 工作线程只消费已预留的任务，不再修改全局预算；
- `ThreadPoolExecutor` 的 `max_workers` 直接来自配置；首波任务通过一次性启动栅栏保障真实并发，报告中记录实际峰值，峰值不超过配置。

## 7. 阶段快照与恢复

阶段固定为 `load -> normalize -> fuse -> truncate`。提交过程：

1. 在 `state/snapshots/.tmp-vNNNNNN-token/` 写入四个阶段 JSON 和 `config_snapshot.json`；
2. 对文件和目录执行 `fsync`；
3. 写入包含阶段版本、阶段 SHA-256、配置哈希、结构哈希、已完成通道、量化参数的 `manifest.json`；
4. 将临时目录重命名为正式版本目录；
5. 用临时文件 + `os.replace` 原子切换 `state/CURRENT`。

因此强制终止只会留下不可见的 `.tmp-*` 目录；启动只读取 `CURRENT` 指向的完整版本。

启动校验包括：

- `CURRENT` 名称、manifest schema 和 `run_version`；
- 配置快照哈希与 manifest `config_hash`；
- 四个阶段文件存在且 SHA-256 一致；
- 每个阶段版本等于 `run_version`；
- 可选 `--expected-version`。

任一不一致产生 `E_SNAPSHOT_STALE` 并放弃部分快照、从当前碎片全量重建。正常配置编辑与上一快照配置不同不视为陈旧，这正是增量运行场景。

## 8. 故障注入

独立注入文件位于 `samples/injections/`。注入器修改真实输入对象后再走正常校验路径，不直接伪造失败记录。

| 注入 | 目标阶段 | 真实动作 | 业务码 |
|---|---|---|---|
| `nonfinite_score` | load | 将目标候选得分替换为 `NaN` | `E_NONFINITE_SCORE` |
| `duplicate_candidate` | load | 在输入流中插入重复候选行 | `E_CONFIG_INVALID` |
| `trunc_limit_out_of_range` | config | 将 `trunc_k` 设置为 `-3` | `E_CONFIG_INVALID` |
| `zero_capacity` | config | 将目标通道容量设置为 `0` | `E_CONFIG_INVALID` |

报告同时记录 `injections_hit` 和失败项中的 `injection_id`，证明命中与业务码映射。

## 9. 数值可观测

独立 float64 参考路径使用与主路径相同的原始分数和归一化配置，但不经过定点化。报告输出：

- 截断边界分数；
- 边界邻域 `boundary_neighborhood`；
- 邻域最大绝对误差 `max_abs_error_at_boundary`；
- `quant_bits`、`quant_step`、`normalization`、`rounding`；
- 实际定点整数校验和、融合分数校验和、实际保留 ID。

切换归一化或量化档位时，校验和证明内部整数、融合分数和截断结果均发生变化，而不是只改标签。

## 10. 本地样本与编辑序列

固定种子为 `20261122`。初始通道候选数：

| 通道 | 初始候选数 | 容量 |
|---|---:|---:|
| `ch_alpha` | 55 | 80 |
| `ch_beta` | 40 | 70 |
| `ch_gamma` | 27 | 60 |
| `ch_delta` | 12 | 40 |

编辑序列：

| ID | 编辑 |
|---|---|
| `E01` | 向 `ch_alpha` 追加 6 个候选碎片 |
| `E02` | 替换 `ch_beta` 的整个候选碎片 |
| `E03` | `ch_alpha` 权重由 `1.5` 改为 `2.5` |
| `E04` | 截断上限 `20 -> 14` |
| `E05` | 归一化 `minmax -> zscore` |
| `E06` | 截断上限 `14 -> 12` |
| `E07` | 量化档位 `8 -> 12` |

## 11. 实际验证结果

完整结果见 `artifacts/acceptance/summary.json`。本次实际运行状态为 `passed`。

### 增量/全量等价

- 7 个编辑步骤的融合列表与顺序、贡献分解、候选/融合指纹、失败清单、归一化事件、观测字段均与空白全量重建一致；每步 `core_mismatches` 为空。
- 最终量化切换后，140 个候选和 73 个融合全部失效，因此最终 `invalidation/reuse` 清单与预算审计也与空白全量逐字段一致。
- 无编辑重跑：134 个候选、67 个融合全部复用，预算消耗 `0`。

关键增量步骤：

| 编辑 | 失效候选 | 复用候选 | 失效融合 | 复用融合 | 增量预算 | 全量预算 |
|---|---:|---:|---:|---:|---:|---:|
| E01 追加 alpha | 61 | 79 | 61 | 12 | 116 | 140 |
| E02 替换 beta | 40 | 100 | 40 | 33 | 105 | 140 |
| E03 改权重 | 0 | 140 | 61 | 12 | 116 | 140 |
| E04 改截断 | 0 | 140 | 73 | 0 | 140 | 140 |
| E05 改归一化 | 140 | 0 | 73 | 0 | 140 | 140 |
| E06 改截断 | 0 | 140 | 73 | 0 | 140 | 140 |
| E07 改量化 | 140 | 0 | 73 | 0 | 140 | 140 |

### 量化/归一化真实性

- `minmax/q8 -> zscore/q8`：定点整数校验和、融合校验和、实际 top-k 内容均改变；
- `zscore/q8 -> zscore/q12`：`quant_step` 从 `0.00390625` 变为 `0.000244140625`，定点整数、融合分数和截断结果均改变。

### 预算调度

| 场景 | 全局预算 | 通道配额 | 消耗 | 剩余 | 待处理贡献 | 失败码 | 实际并发峰值 |
|---|---:|---:|---:|---:|---:|---|---:|
| 配额 20 | 100 | 20 | 72 | 28 | 62 | `E_CHANNEL_QUOTA_EXHAUSTED` | 4 |
| 全局 50 | 50 | 10000 | 50 | 0 | 84 | `E_GLOBAL_BUDGET_EXHAUSTED` | 1 |
| 配额 0 | 10 | 0 | 0 | 10 | 134 | `E_CHANNEL_QUOTA_EXHAUSTED` | 0 |

上述场景重复运行两次，`audit/consumed/pending_queue/remaining` 完全一致，且消耗与授予均未超限。

### 故障注入

| ID | 类型 | 命中业务码 | 隔离结果 |
|---|---|---|---|
| `INJ-001` | 非有限得分 | `E_NONFINITE_SCORE` | 跳过坏候选，其余继续融合 |
| `INJ-002` | 重复候选 | `E_CONFIG_INVALID` | 保留首个，其余继续融合 |
| `INJ-003` | 截断越界 | `E_CONFIG_INVALID` | 配置阶段中止，不提交快照 |
| `INJ-004` | 容量为零 | `E_CONFIG_INVALID` | 配置阶段中止，不提交快照 |

### 快照与恢复

- 强制终止在 `normalize` 后：`CURRENT` 仍指向上一完整版本，临时目录被忽略；
- 下次启动版本正常递增、无陈旧标记，全部已有产物复用；
- 篡改已提交的 `load.json` 后：检出 `E_SNAPSHOT_STALE` 并全量重建；
- `--expected-version` 不符：检出 `E_SNAPSHOT_STALE`；
- 普通配置编辑后仍可续跑，不误报陈旧。

### 自动化测试

```text
Ran 8 tests in 2.161s
OK
```

测试覆盖：四类注入、提交后加载注入不复用坏候选、零分母归一化、单碎片局部失效/复用、预算稳定与上限、崩溃恢复/陈旧检测、最终增量与空白全量等价、`trunc_k=0`。
