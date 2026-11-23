# 技术说明

## 1. 处理管线

```text
加载校验 -> feature -> emission -> decode -> merge
              ^          ^          ^         ^
           样本+词表   嵌入+发射   转移权重  类型证据+容差
```

每阶段产物都是内容寻址的 `StageArtifact`：

```text
stage, sample_id, cache_key, input_fingerprint,
output_fingerprint, payload, refs, version
```

业务结果只由最终语料、词表、参数包和有效配置决定；线程调度与词表插入顺序不进入输出顺序。

## 2. 校验层

### 语料

逐行解析后检查：

1. 对象、非空字符串 ID。
2. 字序列与 BIESO 标签长度相等且非空。
3. 标签属于 `B/I/E/S/O`。
4. 非 `O` 位置类型映射到 `LOC/ORG/PER/TIME`。
5. 每个实体从 `B` 开始；允许 `B E` 或 `B I* E`；孤立 `I/E`、未闭合跨度、跨度内类型变化均拒绝。
6. 重复 ID 的后续行拒绝，首个合法行保留。

非法行进入 `E_CORPUS_INVALID` 失败清单且不生成可推理样本。

### 模型与命令

模型检查正整数维度、嵌入向量长度、发射矩阵形状、标签集合、转移矩阵形状、起始权重和词表类型。命令检查枚举、目标字段和调整权重字段。结构性非有限值不被加载层误判，而是在发射阶段按 `E_EMISSION_INVALID` 处理。

## 3. 特征哈希与定点发射

对位置 `i` 构造五个通道：

```text
cur:<字>
prev:<前字或 @BOS>
next:<后字或 @EOS>
rel:<single|start|middle|end>
hit:<词表最长覆盖长度>
```

每个通道使用 SHA-256 得到：

```text
feature_index = H("fi|" + token)[0:8] mod feature_dim
bucket        = H("fb|" + token)[0:8] mod hash_buckets
```

嵌入查找优先级：通道精确键、字符键、`@bucket:N`、确定性伪随机向量。找到的 `feature_dim` 维向量按 `feature_index` 旋转后累加到字符嵌入。

给定 `S = 10^fixed_decimals`、量化档 `Q`，所有嵌入与发射权重先转为 `S` 定点整数，乘积为 `S²` 单位，再精确除以 `S` 并按 `Q` 取整。取整采用“半值远离零”。每个位置记录：

- 每个标签的定点残差分子/分母和舍入方向；
- 嵌入聚合残差；
- 发射权重残差；
- 最终 `emission_scores`。

因此相同输入的重跑结果不存在浮点顺序漂移。

## 4. 对数域维特比

标签编码固定为 `B=0, I=1, E=2, S=3, O=4`。转移与起始权重按同一配置定点化。前向递推：

```text
score[0][j] = emission[0][j] + start[j]
score[i][j] = emission[i][j] + max_k(score[i-1][k] + transition[k][j])
```

扫描顺序固定为位置升序、目标标签编码升序、前驱标签编码升序。并列时保留最小前驱编码；最终并列时选择最小标签编码。`tie_records` 记录并列标签、选中项和阶段。产物包含每个位置五个标签的最佳路径得分、前驱以及位置候选边际。

## 5. 片段修复与类型归属

固定修复规则：

| 现象 | 修复 |
|---|---|
| 孤立 `I` | 改写为 `B`，记录 `I_WITHOUT_B` |
| 孤立 `E` | 视为 `S`，记录 `E_WITHOUT_B` |
| `B` 后不是 `I/E` | `B` 视为单字片段，记录 `B_WITHOUT_CONTINUATION` |
| 开放片段遇到 `O/S/B/EOS` | 在当前位置前闭合，记录 `ENTITY_NOT_CLOSED_BY_E` |
| 未知标签 | 视为 `O`，记录 `UNKNOWN_LABEL_AS_O` |

类型候选证据：

```text
同一 span 的金标类型           +3
跨度内每个字的本样本金标类型   +1/字
完全匹配的受控词表条目类型     +2/条
```

按得分降序、类型编码 `LOC < ORG < PER < TIME` 升序裁决。并列写入 `type_tie`。

片段置信度取跨度内所有位置“最佳候选减次佳候选”的最小定点边际。若它恰好等于配置容差，记录边界对象：

- 当前分数、容差及其量化残差；
- `fixed_decimals + 1` 参考分数与精确差值；
- 当前/参考路径和类型裁决；
- 最近相邻量化档探测；
- 若决策变化，`flip_stage` 标为 `decode` 或 `merge`。

## 6. 缓存键与精确失效

| 阶段 | 键依赖 |
|---|---|
| feature | 算法版本、样本 ID、字符、样本命中词表文本、`feature_dim`、`hash_buckets` |
| emission | feature 输出指纹、实际使用的嵌入键/向量、发射矩阵、定点位宽、量化档 |
| decode | emission 输出指纹、转移与起始权重、定点位宽、量化档、位置数 |
| merge | decode 输出指纹、样本金标标签/类型、相关词表条目、容差、量化档、参考位宽 |

由此得到：

- 非命中词表增删：所有阶段复用。
- 词表仅改类型：feature/emission/decode 复用，命中样本 merge 重算。
- 单样本仅改标签/类型：前三阶段复用，仅该样本 merge 重算。
- 发射权重调整：feature 复用，emission 及其下游重算。
- 转移权重调整：feature/emission 复用，decode 及 merge 重算；单字样本不使用转移边，可继续复用 decode。
- 定点/量化调整：feature 复用，emission、decode、merge 重算。
- 相同内容的命令不会产生新计算。

实际执行状态写入 `reuse_recompute_ledger`，每阶段带稳定 `task_sort_key = "<sample_id>:<stage>"`。

## 7. 并发与预算语义

- 线程池最大并发严格取 `thread_pool_workers`。
- 每个样本一个任务，提交顺序按样本 ID 升序；最终汇总再次按样本 ID 排序。
- 每个任务内部仍按 feature -> emission -> decode -> merge 顺序执行。
- 内容缓存用锁保护，任务完成顺序不影响读取。
- 命令返回时若本轮最新状态已完成，返回最新摘要；否则复制最近一次已就绪摘要，附加当前受影响清单、进度、`stale=true` 和 `E_LATENCY_BUDGET_EXCEEDED`。
- 后台任务可继续完成；进入下一命令或 `settle()` 时先收割已完成结果。最终稳定态必须完成全部缺失阶段。

线程池在引擎初始化时预热，避免首个命令承担建线程成本。

## 8. 快照协议

目录布局：

```text
artifacts/snapshots/<run_id>/
  CURRENT                     # 原子替换的指针文件
  gen-000001/
    manifest.json
    state.json
    stages/feature/*.json
    stages/emission/*.json
    stages/decode/*.json
    stages/merge/*.json
```

提交过程：

1. 优先在配置 `temp_dir`（解析后须位于工作区内）创建 `.staging-*`；否则在 `gen-NNNNNN` 同级的 `.tmp` 下创建。
2. 比较旧 manifest 的状态哈希与各阶段输出版本。
3. 未变化阶段文件用硬链接（失败时复制）复用；变化阶段重新写。
4. 写 state/manifest，`fsync` 临时文件。
5. 将 staging 目录整体改名为新代次；正式产物只出现在 `snapshots/<run_id>/gen-*`，staging 不参与正式清单。
6. 写临时 CURRENT 文件并 `os.replace` 原子切换指针。
7. 清理三代以前的目录。

清单字段包括 `generation`、`global_version`、`model_version`、`config_hash`、`corpus_hash`、`vocabulary_hash`、`completed_samples`、`stage_versions`、`stage_cache_keys`、`changed_stages`、`state_hash`、`command_position`、`processed_commands`、`reason`。

恢复时校验清单结构、状态哈希、模型版本、配置/语料/词表哈希、完成样本清单和每个阶段文件版本。任何不一致都产生 `E_SNAPSHOT_INVALID` 并从输入重建，不会把损坏快照当成有效缓存。

## 9. 信号与退出

`SIGINT`/`SIGTERM` 处理器只设置停止标记。当前命令仍会生成首响摘要，随后等待稳定、提交快照、写报告，再按 `128 + signal` 返回。`--max-commands` 用于无信号的确定性中断模拟。

## 10. 增量/全量等价投影

`business_projection()` 比较：

- 样本顺序；
- 实体片段及类型；
- 路径标签与总分；
- 逐位置候选得分；
- 修复与边界记录；
- 发射量化细节；
- 各阶段输出指纹与缓存键；
- 当前阶段失败码；
- 模型/语料/词表身份，以及只含固定小数位、量化档、容差的“结果配置哈希”。

运行参数（预算、线程数、快照间隔、临时路径）和 `reuse/recompute` 动作状态属于诊断信息，不进入结果身份。全量空白缓存天然会把动作标为 `recompute`，而增量可能标为 `reuse`；两者比较的是内容键和输出版本，业务数据完全一致。

## 11. 扩展点

- 新增业务码集中放在 `ner_editor/constants.py`。
- 新增命令需同时修改命令校验、执行器与失效影响列表。
- 新增阶段需扩展 `STAGE_ORDER`、缓存键、快照目录和报告账本。
- 模型标签目前固定为五类 BIESO；新增标签类型需要同步维特比和修复规则。
