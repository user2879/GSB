# GSB 确定性样本账本流水线设计

本实现把本地确定性 JSONL 样本依次送入清洗、分片近重复去重、定点加权采样和配比汇总，并为每条结论保存内容指纹、配置版本、上游引用和缓存决策。所有随机样本均由 `gsb/fixtures.py` 使用固定整数种子 `20261127` 生成，不访问网络、外部账号或服务。

## 快速命令

```bash
python3 --version                 # 需要 3.10+
python3 -m pip install -r requirements.txt
python3 scripts/generate_fixtures.py
python3 -m gsb.cli run \
  --config configs/pipeline.json \
  --samples data/batches \
  --state state \
  --workers 2

python3 -m pytest -q tests/test_pipeline.py
python3 scripts/verify_acceptance.py
```

可选诊断界面需要系统 Tk：

```bash
sudo apt install python3-tk tk8.6
python3 -m gsb.cli ui --state state
```

当前验收环境缺少 `libtk8.6.so`，因此自动验收把 Tkinter 标记为“可选组件不可导入”；核心流水线不依赖 Tkinter。

## 目录结构

- `gsb/config.py`：配置枚举、定点口径、权重、目标配比、并发和预算校验。
- `gsb/samples.py`：逐批 JSONL 解析、全局样本唯一性和字段宽度校验。
- `gsb/edits.py`：编辑命令流校验与内存/文件树应用。
- `gsb/fixtures.py`：固定种子样本、配置和编辑命令生成器。
- `gsb/fixedpoint.py`：允许的小数位宽 `3/6/9/12` 与量化档位 `floor/half_up/ceiling`。
- `gsb/cleaning.py`：字典序清洗、缺失位图规则、词项下限、批次内填补和长度归一。
- `gsb/features.py`、`gsb/dedup.py`：子词集合、定点 Jaccard、阈值边界、分片和代表选择。
- `gsb/sampling.py`：定点 D’Hondt 风格加权配额/采样和全局稳定序位。
- `gsb/summary.py`：采样数、去重后占比、有效词项和定点偏差方向。
- `gsb/ledger.py`、`gsb/canonical.py`：规范化 JSON、SHA-256 指纹和账本条目。
- `gsb/store.py`：阶段快照、临时目录、`current` 符号链接原子替换、清单哈希和收尾报告。
- `gsb/engine.py`：缓存复用、线程池、预算、领域配额、待办队列和恢复编排。
- `scripts/verify_acceptance.py`：完整验收矩阵并写出 `artifacts/acceptance_report.json`。
- `tests/test_pipeline.py`：边界相似度、并列决断、校验、预算恢复和并发测试。

## 输入模式

每个批次是一个 JSONL 文件，首行必须是 `batch_header`，随后为样本：

```json
{"record_type":"batch_header","batch_id":"alpha-b0","declared_fields":["token_counts","length_distribution","missing_bitmap"],"impute_values":{"length_distribution":[3,3,4,4,5,5]}}
{"record_type":"sample","sample_id":"alpha-s03","batch_id":"alpha-b0","domain":"alpha","token_counts":{"sw-1":8},"length_distribution":[2,4,1,6,3,5],"missing_bitmap":[0,1,1,1,0,0]}
```

加载器拒绝以下情况并返回 `E_INPUT_INVALID`，错误对象包含 `field`，适用时包含 `batch_id`：

- `sample_id` 跨批次重复；
- 词项计数不是非负整数；
- `length_distribution`、`missing_bitmap` 或配置位宽不一致；
- 位图值不是 `0/1`；
- 领域不在 `config.domains` 声明集合；
- 权重为负、NaN/Infinity 或权重总和不为正；
- 千分比阈值不是 `0..1000` 的整数；
- 分片大小、并发、预算、配额非法；
- 定点位宽或量化档位不在允许枚举；
- 目标配比定点量化后不精确合计为一个规模单位。

系统不会为缺失的必填配置静默填入默认值。

## 确定性规则

### 清洗

- 全量样本按 `sample_id` 字典序形成稳定结论顺序。
- 配置中的缺失规则按顺序匹配位图位置；命中即记录规则码、样本字段和 `config_field`。
- 词项总量低于 `cleaning.min_total_token_count` 时记录 `LOW_TOKEN_COUNT`。
- 位图为 `1` 的长度箱只使用该样本所属批次头中的 `impute_values.length_distribution` 填补，不读取后续批次。
- 长度分布用最大余数法归一到 `10**decimal_width` 个定点单位，保证各箱之和精确为规模单位。

### 去重

- 每个领域内按样本标识排序并按 `dedup.shard_size` 切分片。
- 正计数词项的规范化键构成子词集合。
- 相似度为 `intersection / union`，在配置定点位宽下量化为整数 `similarity_fp`；阈值转换为 `threshold_permille * 10**(width-3)`。
- 同时保留精确有理判定信息：`intersection * 1000 >= threshold * union`。当 `intersection * 1000 == threshold_permille * union` 时标记 `exact_threshold_boundary=true`。
- 代表按定点综合分降序、`sample_id` 升序决断。综合分包含词项量、完整度和只引用本分片领域的权重分量；因此调整某领域权重可改变该领域分片的代表排序，但不会改变其他领域分片的缓存键。
- 每个被剔除样本记录 `representative_sample_id`、`similarity_fp`、千分比相似度和边界标记。

### 采样与配比

- 采样使用定点权重比较 `weight_fp / (allocated + 1)`，逐席分配；比值完全相同时选择下一个字典序最小的候选样本。
- 每条中选结论有从 `0` 开始的 `global_rank`。
- 汇总输出每领域采样数、采样占比定点值、去重后样本数与占比、采样样本有效词项总量。
- `deviation_fp = sampled_ratio_fp - target_ratio_fp`，方向为 `above_target`、`below_target` 或 `equal`。

## 指纹与精确失效

缓存键均由规范化 JSON 的 SHA-256 派生：

- 清洗：样本内容指纹、清洗相关配置指纹、定点位宽/档位和算法版本。
- 分片去重：有序成员及清洗结论指纹、阈值/分片/评分配置、该分片引用领域的权重分量和量化口径；每个簇还派生 `cluster_cache_key`。
- 采样：存活样本集合、去重产物指纹、全部领域权重/采样配置和量化口径。
- 汇总：采样产物、去重产物、目标配比和量化口径。

账本条目包含 `ordinal`、`entry_id`、`config_version`、完整 `config_hash`、相关 `scope_fingerprint`、`cache_key`、`upstream_fingerprints`、`artifact_fingerprint`、`disposition` 和失效依据。修改一个领域权重时：

- 清洗缓存键不变，全部复用；
- 仅引用该领域权重的分片去重簇重算，其他领域分片命中持久化产物；
- 采样序位和配比汇总重算。

替换单条样本时，只有该样本的清洗结论、包含它的分片以及下游采样/汇总失效；其他五个分片复用。若新增/删除样本导致分片成员边界变化，成员指纹变化的后续分片也会按指纹精确失效，这是预期行为。

冷启动全量重建没有旧产物，因此所有条目的执行态都是 `recomputed/cold_build`；热增量运行会有 `reused`。验收比较的是业务字段、缓存键、上游引用、制品指纹和规范账本顺序；同时单独比较任意预算切片/并发路径下相对同一基线的 `reused/recomputed` 执行清单。

## 预算、配额与并发

调度成本是确定的整数工作单位：

- 每条清洗样本：`1`；
- 每个去重分片：`1 + C(n,2)` 次比较；
- 采样和汇总：各 `1`。

运行时 `--budget N` 或 API 的 `budget=N` 限定本轮可消耗单位。预算不足时，未完成项按稳定顺序写入 `queue.items`，清单状态为 `E_BUDGET_PENDING`；下一轮发现同一输入/配置版本时优先依据已提交队列续跑。

领域配额使用每领域 `quota`，未声明时回退到 `budgets.domain_quota`。超过配额立即形成 `E_DOMAIN_QUOTA_EXCEEDED`，错误定位到 `domains.<domain>.quota` 和当前批次。分片比较对数超过 `budgets.shard_comparison_limit` 时返回 `E_SHARD_COMPARISON_LIMIT`。

清洗和去重分别使用 `ThreadPoolExecutor`，worker 数取配置并受 `concurrency.max_workers` 与运行时 `--workers` 的最小值约束。所有共享结果均为不可变输入上的纯函数，最终按样本/分片稳定排序合并，所以并发度不改变结论。

## 快照与强制终止

`SnapshotStore.commit()` 将 `manifest.json`、`cleaning.json`、`dedup.json`、`sampling.json`、`summary.json`、`ledger.json`、`queue.json`、`budget.json` 和 `error.json` 写入 `state/tmp/...`，fsync 后原子改名为 `state/snapshots/snapshot-NNNNNN`，再通过临时符号链接和 `os.replace` 原子替换 `state/current`。

提交前使用 `commit.lock` 重新读取当前清单，并比对 `parent_global_version`；版本不符返回 `E_VERSION_CONFLICT`。清单保存各组件 SHA-256，加载时校验，而不是逐文件覆盖或信任临时目录。

`--crash-before-commit` 在临时快照落盘并 fsync 后、原子切换 `current` 前向自身发送 `SIGKILL`。重启只读取上一个已提交的 `current`，忽略未提交临时目录，并依据清单中的阶段状态和队列续跑。可处理的正常退出可通过 `gsb.cli write-closeout` 或信号处理写出 `state/reports/closeout-latest.json`；SIGKILL 不承诺收尾报告。

## 已验证属性

完整结果见 `artifacts/acceptance_report.json` 与 `docs/VERIFICATION_RESULTS.md`。验收覆盖：

- 13 类输入/配置违规的业务码与字段定位；
- 5 个编辑前缀的增量运行与空白全量重建逐字段一致；
- 权重和单样本替换的精确复用/重算集合；
- `0..total_cost-1` 每个预算耗尽点后续跑与串行结果一致；
- worker `1..4` 结果一致且不超过配置并发；
- 热增量在不同预算切片后的复用/重算清单一致；
- `E_BUDGET_PENDING`、`E_DOMAIN_QUOTA_EXCEEDED`、`E_SHARD_COMPARISON_LIMIT`；
- 真实 SIGKILL 后恢复结果与独立空白全量一致。
