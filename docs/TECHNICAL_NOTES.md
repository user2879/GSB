# 类别特征交叉编码诊断组件说明

## 1. 交付内容

- `category_cross_diagnostic/config.py`：YAML 读取、模式校验、稳定错误码与失败清单。
- `category_cross_diagnostic/sample_data.py`：固定整数种子的确定性表格、缺失位图、列内容指纹和定点量化。
- `category_cross_diagnostic/enumeration.py`：稳定组合顺序、内容/阶数指纹、位宽作用域指纹。
- `category_cross_diagnostic/encoding.py`：盐化 BLAKE2b 哈希、位掩码编码、空值、真实基数和碰撞统计。
- `category_cross_diagnostic/budget.py`：基数预算、哈希桶降级/跳过、编码点数和双重裁决账本。
- `category_cross_diagnostic/flip_analysis.py`：编码位宽与定点/量化档位相邻重算、翻转和来源归因。
- `category_cross_diagnostic/cache.py`：内容寻址阶段缓存与逐条复用账本。
- `category_cross_diagnostic/snapshot.py`：临时目录写入 + 目录提交 + `latest.json` 指针的原子快照。
- `category_cross_diagnostic/engine.py`：串行/线程池执行、精确失效、预算截止、恢复和 JSON 报告。
- `scripts/validate_acceptance.py`：本地样本端到端验收。
- `scripts/fuzz_sequences.py`：12 组确定性随机编辑序列一致性压测。
- `tests/test_diagnostic.py`：7 个标准库 `unittest` 用例。

运行时只使用 Python 标准库、NumPy 和 PyYAML，不访问网络或外部数据集。

## 2. 快速运行

在仓库根目录执行：

```bash
python3 -m pip install numpy pyyaml
python3 scripts/generate_sample_data.py
python3 -m category_cross_diagnostic.cli validate \
  --features examples/feature_table.yaml \
  --strategy examples/strategy.yaml
python3 -m category_cross_diagnostic.cli run \
  --features examples/feature_table.yaml \
  --strategy examples/strategy.yaml \
  --state-dir artifacts/cli_state \
  --report artifacts/cli_diagnostic_report.json
python3 -m unittest discover -s tests -v
python3 scripts/validate_acceptance.py
python3 scripts/fuzz_sequences.py
```

Windows PowerShell 下将行尾续行符改为反引号，或直接写成单行；脚本会自行把仓库根加入 `sys.path`。

## 3. 枚举语义

候选列对按无向、有序列名对归一化。重复对和自交叉输出 `E_DUPLICATE_PAIR`，随后去重，不产生两个等价组合。

组件支持两种展开模式：

- `expansion_mode: universe`（默认）：候选对端点的并集作为合格类别列，枚举所有大小 `1..order_limit` 的子集。这最贴近“全部大小不超过上限的列组合”的字面语义。
- `expansion_mode: clique`（示例使用）：二阶组合必须是显式候选对；`k>=3` 组合只有在其二阶子对均被候选策略支持时才进入。该模式让“增删一条候选边”的影响边界更明确。

所有模式均按组合长度升序、同长按列名字典序元组排序。示例的 11 个组合顺序由验收脚本逐字断言。

每个组合有两层指纹：

- `base_fingerprint = SHA256(order, [(column, column_content_fingerprint), ...])`，只由列内容和组合阶数派生。
- `combo_fingerprint = SHA256("combo-width-scoped-v1", base_fingerprint, width_gear_fingerprint)`，因此切换编码位宽、定点位宽或量化档位会产生新的活动组合指纹和缓存作用域。

## 4. 编码、基数与碰撞

每行严格按行序号升序处理。参与列的缺失状态和带类型值先转成规范 JSON，再用由 `base_fingerprint + fixed_point_width + quantization_levels` 派生的每组合盐值隔离编码空间。编码整数为：

```text
BLAKE2b(canonical_tuple, salt) mod 2 ** encoding_width
```

编码位宽不进入盐，因此相邻编码位宽是在同一完整哈希上增加/减少低位掩码；宽档位只会细分窄档位的等价类，使“编码空间压缩”归因可解释。定点/量化档位进入盐，用来模拟不同位宽档位的真实空间隔离和再分布。

每个组合输出：

- `true_unique_count`：未哈希的真实取值元组数。
- `unique_code_count`：实际编码字典大小。
- `null_row_count`：至少一列缺失的行数。
- `null_code_count`：缺失元组映射到的编码数。
- `collision_count`：`true_unique_count - unique_code_count`，即被合并的不同取值元组超额数。
- `colliding_bucket_count`、`colliding_pair_count`：碰撞桶数和精确配对碰撞数。
- 排序后的字典桶摘要与 `dictionary_digest`。

## 5. 双重预算裁决

点数成本固定为：

```text
point_cost = row_count * encoding_width
```

组件按稳定组合序先完成编码/基数统计和基数裁决，再计入点数。这样边界组合即使同时触发两类约束，也不会因为点数预检而缺少基数结论。

基数超预算时：

- `over_budget_action: skip`：输出 `E_CARDINALITY_BUDGET_SKIP`，组合不保留。
- `over_budget_action: truncate`：按 `truncate_buckets`（缺省等于基数预算）做确定性哈希桶截断，记录截断前后字典量、桶数和移除条目数。

当某组合计入点数后超过 `point_budget`：

- 该组合仍保留完整裁决，并写入 `E_POINT_BUDGET_EXHAUSTED`。
- 后续组合进入 `pending`，顺序与组合枚举顺序一致。
- `decision_sequence` 固定为：1）基数预算；2）点数预算。
- `joint_constraint_triggered=true` 表示同组合双触发。
- `dominant_constraint` 按本地结果严重性判定：跳过为 `cardinality_budget`；否则点数耗尽为 `point_budget`；截断为 `cardinality_budget`。
- `terminal_constraint=point_budget` 表示点数耗尽负责终止后续调度。

验收中的 `point_budget: 90000` 场景稳定得到边界组合 `cross__channel__city__device` 双触发，唯一挂起组合为 `cross__city__device__membership`。

## 6. 位宽翻转归因

`flip_attribution` 对每个组合重算两条相邻档位链：

- 编码位宽链：`8 -> 12 -> 16 -> 24 -> 32 -> 64`，定点/量化参数保持不变。
- 定点量化链：`f8/q8 -> f16/q16 -> f32/q32`，编码位宽保持不变；若当前配置不在标准档中，会插入当前档。

每对相邻档位输出前后活动指纹、编码样本、唯一数、碰撞数、裁决签名和是否翻转。来源贡献包括：

- `encoding_space_compression`：码空间、空间压力及其变化。
- `collision_count_change`：碰撞超额、碰撞桶和配对碰撞增量。
- `budget_boundary`：基数预算余量、点数预算余量及是否跨越裁决边界。
- `fixed_point_quantization_profile`：定点/量化盐切换造成的再分布贡献。

样本中的三列组合 `cross__channel__city__device` 从 `e8` 到 `e16` 的唯一编码数为 `225 -> 543`，碰撞数为 `321 -> 3`，活动指纹、字典摘要和编码样本均真实变化。

## 7. 精确失效与缓存键

四个阶段分别生成缓存键：

```text
cache_key = SHA256(
  "cache-key-v2",
  stage,
  active_combo_fingerprint,
  base_fingerprint,
  stage_parameter_fingerprint,
  width_gear_fingerprint
)
```

阶段参数按依赖最小化：

- 枚举：展开模式、阶数和该组合的候选对支持。
- 字典：哈希算法、规范格式、预览策略和编码参数。
- 基数：真实元组/碰撞/空值语义。
- 裁决：两类预算、降级策略、稳定序号和该序号下的累计点数。

因此：

- 改基数预算只会使所有 `budget_decision` 键失效，枚举、字典和基数诊断直接复用。
- 改点数预算或在前面插入组合会使裁决序号/累计点数变化，所以下游裁决失效；编码和基数仍复用。
- 替换单列数据只改变引用该列的组合内容指纹与活动指纹；不含该列的组合四阶段全部命中。
- 切位宽会改变 `width_gear_fingerprint`，从而使位宽相关阶段按预期重算。

报告中的 `cache_ledger` 逐条给出 `cache_key`、阶段、组合指纹、基础指纹、参数指纹、位宽指纹、列内容指纹和稳定任务键，可对照 `state/cache.json` 复核。

需要区分执行历史与诊断产物：冷启动全量的真实缓存状态必然全部是 `recomputed`，增量运行则会有 `reused`。因此“增量 vs 空白全量逐字段一致”比较的是所有确定性产物、缓存键、指纹、裁决、挂起顺序和失败清单；真实命中状态属于执行历史。为验证复用/重算清单本身的确定性，验收额外从空白重放同一条编辑序列，两次最终 `cache_ledger`（含状态）逐字段相同。

## 8. 并发模型

`DiagnosticEngine(serial=False)` 使用 `ThreadPoolExecutor(max_workers=strategy.max_workers)`。需要编码的任务先按：

```text
stable_task_key = SHA256(stage_name + combo_fingerprint)
```

排序后提交；结果回到主线程按组合序写入缓存和账本。执行期用锁维护活动任务数，报告输出配置并发数和实际观测峰值，且峰值不会超过配置值。

验收样本配置为 4，实测 `observed_max_active_tasks=4`；串行/并行冷启动报告在规范化运行时观测值后逐字段一致。

## 9. 快照与恢复

`SnapshotStore.commit()` 先在 `snapshots/.staging-*` 临时目录写入：

- `manifest.json`
- `enumeration.json`
- `dictionary.json`
- `cardinality.json`
- `decisions.json`
- `pending.json`
- `processed_combinations.json`
- `cache_ledger.json`

文件 fsync 后将临时目录整体 `os.replace` 为不可变 `snapshot-NNNNNN`，再原子替换 `latest.json` 指针。强制终止时只可能留下旧快照或临时目录，已提交快照是最近一次完整整体替换。

清单记录：

- `global_version`
- `policy_config_hash`
- `table_fingerprint`
- `width_gear_fingerprint`
- `completed_combo_ids`
- `consumed_points`
- 总组合数和点数预算

启动时若版本、策略哈希、表指纹或位宽指纹不一致，判定 `rerun`；全部一致才 `resume`。协作式停止信号（`SIGINT`/`SIGTERM`，Windows 还尝试 `SIGBREAK`）只设置事件，引擎在进入下一组合前检查，因此不会在单个组合编码/统计中途写半成品；到达提交间隔时整体落盘。验收用提交后请求停止的方式证明恢复结果与未中断全新串行结果无字段差异。

## 10. 实际验证结果

以下结果来自当前代码在本机的实际运行：

- `python3 -m unittest tests.test_diagnostic -v`：7/7 通过，记录在 `artifacts/unittest_result.txt`。
- `python3 scripts/validate_acceptance.py`：退出码 0，`all_checks_passed=true`，记录在 `artifacts/acceptance_validation.json`。
- 增量最终态与空白全量：`incremental_vs_blank_full_field_match=true`，前 10 个差异为空。
- 串行与并行：`serial_vs_parallel_field_match=true`，前 10 个差异为空。
- 双次重放编辑序列：`twin_sequence_reuse_ledger_match=true`。
- 最终单列替换：仅 `cross__membership` 和 `cross__city__membership` 两个仍存在的依赖组合重算；8 条阶段记录重算，36 条复用，依赖集合精确匹配。
- 并发：配置 4，实测峰值 4，未超过上限。
- 快照恢复：`matches_fresh_after_recovery=true`。
- 随机编辑序列：`scripts/fuzz_sequences.py` 的 12 组、每组 8 次编辑全部 `all_match=true`，记录在 `artifacts/fuzz_sequence_validation.json`。
- 完整 JSON 诊断报告示例：`artifacts/acceptance_diagnostic_report.json` 与 `artifacts/cli_diagnostic_report.json`。

## 11. 已知边界

- 哈希截断天然可能碰撞；组件不承诺消除碰撞，而是量化并归因碰撞。
- 点数预算的边界组合会被完整诊断后标记耗尽，因此实际 `consumed_after` 可能略高于预算；超出值在 `point.exceeded_by` 中给出。
- 真实缓存命中状态随执行历史变化；确定性等价比较应使用 `comparison.normalize_report(..., cache_status=False)`。冷串行/冷并行可比较包含状态的完整账本。
