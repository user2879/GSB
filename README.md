# 类别特征交叉编码与可复核账本

本项目在本地 YAML + 确定性样本上实现类别列组合枚举、哈希交叉编码、基数/编码点数双重裁决、位宽与量化翻转归因、内容指纹缓存、增量复用账本、并发一致性以及阶段化快照提交。

## 环境自检

已在 WSL2 Ubuntu 环境验证：

```bash
python3 --version
# Python 3.11.0rc1

python3 -m pip install numpy==1.26.4 pyyaml==6.0.1 pytest==8.2.0
# 实测三个依赖均已满足
```

不使用 Docker、网络数据或全局随机状态；样本由 YAML 中的固定整数种子和列名派生种子生成。

## 一键验证

```bash
# 1. 生成确定性 JSON/CSV 样本与缺失位图
python3 scripts/generate_sample.py

# 2. 全量运行默认策略（14 个组合，预算充足）
python3 -m cross_coding.cli run \
  --features config/features.yaml \
  --policy config/policy.yaml \
  --state state/release \
  --report reports/default_report.json

# 3. 运行紧编码点数预算策略，观察 E_POINT_BUDGET_PENDING
python3 -m cross_coding.cli run \
  --features config/features.yaml \
  --policy config/policy_tight_points.yaml \
  --state state/release_tight \
  --report reports/tight_report.json

# 4. 验证三类独立注入故障
python3 -m cross_coding.cli inject

# 5. 运行全部一致性测试
python3 -m pytest -q
```

报告同时输出 JSON 和 Markdown，例如 `reports/default_report.json` 与 `reports/default_report.md`。

## 目录结构

- `cross_coding/config.py`：YAML 载入、规范化和全量校验。
- `cross_coding/table.py`：固定种子数据生成、缺失位图、列内容指纹和定点量化。
- `cross_coding/combos.py`：确定性组合枚举、组合指纹、候选对盐值和稳定任务键。
- `cross_coding/coding.py`：BLAKE2b 确定性哈希编码、碰撞统计、桶截断和点数计划。
- `cross_coding/pipeline.py`：线程池编排、分阶段缓存、双重裁决、翻转归因和对比视图。
- `cross_coding/store.py`：内容寻址缓存、临时目录快照、版本乐观锁和恢复判定。
- `cross_coding/replay.py`：编辑序列应用与增量/空白全量重放辅助。
- `cross_coding/cli.py`：生成样本、运行流水线和注入故障命令。
- `config/features.yaml`：本地特征表描述。
- `config/policy.yaml`：默认交叉策略。
- `config/policy_tight_points.yaml`：点数预算截断演示。
- `config/injections/`：越界值、不存在列、非法位宽三类独立故障。
- `tests/`：校验、枚举、编码、预算、翻转、增量一致性、并发和快照测试。

## 配置与错误码

允许集合为：

- 编码位宽：`4, 8, 16, 32, 64`
- 定点小数位宽：`8, 16, 24, 32, 64`
- 量化档位：`8, 16, 32, 64, 128, 256`

校验会保留全部错误并按校验阶段/配置位置形成稳定失败清单。顶层业务码为 `E_CONFIG_INVALID`，同时给出细分码：

- `E_VALUE_OUT_OF_RANGE`：显式样本值超出类别集合或数值范围。
- `E_UNKNOWN_COLUMN`：候选列对引用不存在列。
- `E_REFERENCE_NOT_CATEGORICAL`：候选列对引用数值列。
- `E_WIDTH_NOT_ALLOWED`：编码位宽不在允许集合。
- `E_FIXED_POINT_NOT_ALLOWED`：定点位宽不在允许集合。

点数预算截断使用 `E_POINT_BUDGET_PENDING`；快照版本乐观锁冲突使用 `E_VERSION_CONFLICT`。

## 组合枚举与指纹

默认 `combination_enumeration: all_subsets`：

1. 从候选列对收集活动类别列；
2. 枚举大小 `1..max_order` 的全部字典序组合；
3. 按组合长度升序，同长度按列名元组字典序排列。

另外支持两种更严格模式：

- `pair_seeded`：二阶只保留候选列对，高阶组合必须至少包含一个候选列对。
- `connected_subsets`：高阶组合必须能通过候选列对图保持连通。

每个组合同时记录：

- `content_fingerprint`：由列名、阶数、参与列内容指纹、可复用候选对及其指纹派生；
- `combo_fingerprint`：在内容指纹上再绑定编码位宽、定点位宽和量化档位；
- `reusable_pairs` / `reusable_pair_fingerprints` / `reusable_pair_salts`；
- 每个阶段的 `task_sort_key = sha256(combo_fingerprint + stage)`。

替换单列数据会改变该列指纹，因此只使引用该列的组合内容指纹变化；增删候选对会改变可复用该对的组合盐值。

## 编码与统计

对每一行的参与列取值（缺失值使用稳定哨兵）做规范化 JSON 序列化，以 `combo_fingerprint` 作为盐值输入 BLAKE2b，再按编码位宽取无符号整数。整个过程不调用运行时随机源。

每个组合保留：

- 逐行 `codes`
- `code_counts` 与编码字典摘要
- `total_count`
- `unique_count`
- `collision_count = total_count - unique_count`
- 浮点 `collision_rate`
- `collision_pair_count`
- 相邻高档位和高一位宽参考统计

## 双重预算裁决

基数预算：

- `unique_count <= cardinality_budget`：`CARDINALITY_RETAINED`
- 超预算且策略为 `truncate_hash_buckets`：映射到 `cardinality_budget` 个桶，结论为 `CARDINALITY_TRUNCATED`
- 超预算且策略为 `skip`，或预算为 `0`：`CARDINALITY_SKIPPED`

编码点数：

- 单组合成本为 `table.row_count * encoding_width_bits`
- 按固定组合序累计；超出 `encoding_point_budget` 的首个组合及其后续组合进入 `pending_queue`
- 待办项逐条携带 `E_POINT_BUDGET_PENDING`、排名、指纹和累计点数
- 下一次持久化运行先读取旧待办键并放入调度优先级，避免从组合序起点重做；最终结果仍归约回固定组合序，保证与空白全量逐字段一致

## 翻转归因

切换编码位宽或量化档位会改变档位绑定的 `combo_fingerprint`、哈希盐和逐行编码整数，而不是只改展示。

每个已处理组合输出相邻档位场景：

- `width_up` / `width_down`
- `quant_up` / `quant_down`

并分别标记：

- `hash_salt_change`：盐值/编码整数/唯一数或碰撞数变化
- `truncation_bucket_count_change`：降级桶数或截断动作变化
- `point_budget_order_change`：位宽导致点数成本和截断排名变化

## 增量缓存与账本

四个阶段均使用内容寻址缓存键：

```text
cache_key = fingerprint(stage, combo_fingerprint, stage_strategy_fingerprint, width_profile)
```

- 列数据/候选对变化：相关组合的枚举、编码、基数和裁决键失效。
- 只改基数或点数预算：编码与基数诊断缓存命中，仅裁决阶段重算。
- 只改位宽/量化档位：档位指纹变化，相关编码及下游失效。
- 未受影响组合显示 `reuse_hit`，当前失效项显示 `recomputed` 并列出 `column_content_changed:<name>`、`candidate_pair_membership_changed`、`policy_parameter_changed:<field>` 等依据。
- 被删除组合记录 `invalidated_removed`。

测试中的 13 步主序列和 3 组确定性伪随机 8 步序列均逐步骤比较增量持久化运行与空白全量运行的 `comparison_view`，覆盖编码整数、唯一数、碰撞数、裁决、待办内容/顺序和业务码计数。

## 快照与恢复

`SnapshotStore` 将 `enumeration`、`coding`、`cardinality`、`adjudication`、`pending_queue`、`ledger` 阶段产物写入不可变快照目录。提交流程为：

1. 读取当前 `global_version` 作为 `expected_version`；
2. 在 `snapshots/.vXXXXXX.tmp-*` 临时目录写入全部 JSON 和清单；
3. 校验提交版本未被其他提交推进；
4. 原子重命名为 `snapshots/vXXXXXX/`；
5. 用 `os.replace` 整体替换根 `manifest.json` 指针。

清单记录：

- `global_version` / `parent_version`
- `policy_config_hash`
- `width_profile`
- `completed_command_cursor`
- `snapshot_commit_interval`
- 各阶段 `artifact_version`、SHA-256 和字节数

根清单损坏或阶段哈希不匹配时，`recover()` 会选择最新的有效不可变快照恢复；过期 `expected_version` 提交抛出 `E_VERSION_CONFLICT`，不会逐文件覆盖新版本。

## 实测结果

最近一次本地验证：

```text
Python 3.11.0rc1 on Linux ...-microsoft-standard-WSL2
numpy==1.26.4, PyYAML==6.0.1, pytest==8.2.0
19 passed in 7.66s
```

默认样本：

- 生成 72 行、6 列（4 类别列 + 2 数值列）及缺失位图。
- 默认全子集/三阶策略枚举 14 个组合。
- 裁决结果为 7 个保留、7 个哈希桶截断、0 个待办。

紧点数策略：

- 每组合成本 `72 * 8 = 576`。
- 预算 `3000` 时前 5 个组合纳入累计，后续 9 个组合稳定进入 `E_POINT_BUDGET_PENDING` 队列。

注入故障：

- `value_out_of_range` -> `E_CONFIG_INVALID` + `E_VALUE_OUT_OF_RANGE`
- `unknown_pair` -> `E_CONFIG_INVALID` + `E_UNKNOWN_COLUMN`
- `invalid_width` -> `E_CONFIG_INVALID` + `E_WIDTH_NOT_ALLOWED`

可复核产物：

- `reports/default_report.json`
- `reports/tight_report.json`
- `reports/injection_report.json`
- `reports/pytest_full_output.txt`
- `state/release/manifest.json`
- `state/release/ledger.jsonl`
