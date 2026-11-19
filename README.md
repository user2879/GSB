# 跨模态确定性检索与一致性验证工具

本项目在本地构建词面/正文共现双通道检索管线，并把索引演进、压缩格式切换、定点融合和中断恢复做成可逐字段复核的确定性流程。实现不依赖外部模型、网络数据或容器环境。

## 交付内容

- `cm_fusion/`：加载、归一化、倒排压缩、v1/v2 转换、检索融合、快照恢复与验证实现。
- `tools/generate_samples.py`：固定种子样本生成器。
- `samples/`：确定性文档流、查询集、融合配置、版本兼容表、编辑序列与最终输入。
- `tests/`：11 项单元/集成测试。
- `reports/demo/`：实际一致性诊断、逐查询结果、索引版本与转换产物。
- `reports/demo/consistency_report.md`：本次验收报告。

## 环境

已使用 Python 3.11 与 NumPy 1.26.4。当前工作区的 `.venv` 是离线可运行环境：

```bash
.venv/bin/python -c 'import sys,numpy; print(sys.version); print(numpy.__version__)'
# Python 3.11.0rc1
# 1.26.4
```

依赖声明见 `requirements.txt`。若系统提供 `python3.11-venv`/pip，可常规创建虚拟环境；本工作区缺少 `ensurepip`，因此按如下方式从本地 NumPy 1.26.4 cp311 wheel 解包安装，未访问网络：

```bash
python3.11 -m venv --without-pip .venv
python3.11 - <<'PY'
import pathlib, zipfile
wheel = pathlib.Path('/path/to/numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl')
zipfile.ZipFile(wheel).extractall('.venv/lib/python3.11/site-packages')
PY
```

## 快速复现

```bash
# 1. 生成固定种子样本
.venv/bin/python tools/generate_samples.py --out samples --seed 20261120

# 2. 运行测试
.venv/bin/python -m unittest discover -v

# 3. 运行完整一致性验收
.venv/bin/python -m cm_fusion verify --samples samples --out reports/demo
```

验收命令会重放 8 条编辑命令、从 `samples/final/` 全量重建，并比较索引指纹、逻辑倒排摘要、块摘要、降级损失、失败清单和每条查询的全部结果字段。

## 输入格式

文档流 `docs.jsonl`：

```json
{"doc_id":"D0001","title_terms":["Alpha","Fusion"],"body_terms":["Index","Snapshot","Fusion"],"shard":"shard-a"}
```

查询集 `queries.jsonl`：

```json
{"query_id":"Q0001","terms":["Alpha","Fusion"],"stream_version":"q1"}
```

融合配置包含：

- `normalization`：大小写、全半角、空白归一化规则。
- `controlled_vocabulary`、`allowed_shards`：受控词表与合法分片。
- `posting_block_size`、`compression_tiers`、`active_compression_tier`：倒排分块与档位 1/2。
- `keep_positions`、`index_format_version`：位置段与 v1/v2 产物选择。
- `channel_weights`、`fusion_temperature`、`fixed_decimal_bits`：定点融合参数。
- `interaction_budget_ms`、`concurrency`、`snapshot_interval_batches`、`top_k`：交互与提交参数。

版本兼容表 `compat.json` 声明 v1/v2 的块头类型、固定字段宽度、编码、档位语义、块级校验和位置跳表能力，以及允许的查询流版本。

## 命令

```bash
# 构建快照并输出结果
.venv/bin/python -m cm_fusion build \
  --state state/demo --docs samples/docs.jsonl --queries samples/queries.jsonl \
  --config samples/config.json --compat samples/compat.json --out out/demo

# 重放编辑流。示例配置预算为 250ms；验收时为避免机器速度影响显式放宽
.venv/bin/python -m cm_fusion edit \
  --state state/demo --command-file samples/edits.jsonl --out out/demo --budget-ms 120000

# 校验并恢复 CURRENT 快照
.venv/bin/python -m cm_fusion resume --state state/demo --config samples/final/config.json

# 将当前快照转换为 v1，并输出降级损失与档位映射
.venv/bin/python -m cm_fusion convert \
  --state state/demo --out out/v1 --target-format v1 --tier 2 --replace
```

预算不足时 `edit` 返回：

```json
{"status":"E_INTERACT_BUDGET_PENDING","completed_stage":"command_received","summary":{"completed_queries":8,"result_count":40,"progress":1.0}}
```

此时不提交新快照，命令可安全重试。超预算命令、耗时和已完成阶段写入 `state/demo/budget_events.jsonl`。

## 设计要点

### 原子快照与恢复

每个版本位于 `snapshots/vNNNNNNNN/`，内部依次包含 `inputs`、`normalized`、`compressed`、`fusion`。各阶段先写 `.<stage>.tmp`，再目录级替换；全部完成后才原子替换根目录 `CURRENT.json`。强制终止只会留下无指针引用的工作目录或快照，`resume` 会清理临时目录并继续使用最后一个完整快照。

`manifest.json` 记录全局版本、配置哈希、兼容表哈希、源文件哈希、已完成批次数、阶段摘要和各阶段版本。`CURRENT.json` 同时校验版本、manifest SHA-256 与配置哈希；不一致返回 `E_CONFIG_MISMATCH`。

### 倒排与两档压缩

文档词项先做大小写折叠、NFKC 全半角折叠和空白合并，再统计标题/正文词频与位置。倒排按 `posting_block_size` 切块，文档序号使用稳定 64 位哈希并做碰撞检测。

- 档位 1：文档序号、词频、位置均采用无符号变长整数差值编码。
- 档位 2：同一无损差值主体叠加低频词过滤；低于阈值的词块进入 `lowfreq` lane，并仍可完整解码。
- `keep_positions=false` 时 payload 不写位置计数和位置值，块头 flags 与块摘要都明确记录 `has_positions=false`。

每个块记录 SHA-256、CRC32（用于 v1 块级校验）、词频和位置计数、lane、格式 flags。整索引指纹包含逻辑倒排摘要、目标格式/档位/位置语义、块摘要、旁路元数据和降级损失，但不含复用统计或时钟。

### v1/v2 转换

v1 使用兼容表驱动的定宽块头与块级 CRC32；v2 使用变长块头和位置跳表。转换时：

- v1→v2：块级校验和移入 `sidecar.json`，记录 `block_checksum` 降级项。
- v2→v1：位置跳表移入 `sidecar.json`，记录 `position_skiplist` 降级项。
- 档位按兼容表映射，例如 `delta_varint_lowfreq_sidecar` ↔ `delta_varint_lowfreq_tail`。
- 转换后重新解码的词频、位置集合与逻辑倒排摘要逐项一致。

### 定点融合

- 词面通道：查询唯一词项被标题覆盖的比例。
- 共现通道：查询与正文在查询词维度上的余弦相似度。
- 每路贡献使用有理数/高精度 Decimal 按 `weight / temperature` 计算，并以 half-up 舍入为 `fixed_decimal_bits` 定点整数；最终分数为两路整数贡献之和。
- 排序按 `(-fused_fixed, doc_id)`，同分使用文档 ID 字典序。
- 同时计算 NumPy `float64` 参考路径，输出每条查询的最大误差单位和超差文档。

### 失效识别与选择性复用

- 格式、压缩档位、位置开关或块大小变化时，旧块格式不可复用，相关块全部重建。
- 权重/温度变化时倒排块全部复用，只重建查询结果。
- 文档追加或修正时，仅内容 SHA 变化的块重建；查询复用签名只覆盖查询相关词项的文档与词频，因此无关查询直接复用。
- 结果包含 `invalidation`：索引失效原因、档位语义映射、复用/重建块和查询清单、查询级失效原因及降级损失。

### 并发与确定性

查询任务携带原始查询序号和 `query_id`，并发完成后按原始序号稳定归并。失败清单按 `(source,line,code,item_id,message)` 排序。并发与单线程比较使用结果顺序、逻辑代价、候选数和确定性耗时；实际墙钟耗时仅作为观测值保留，不参与相等性判定。

## 错误码

- `E_DOC_INVALID`：文档语义错误，如空正文、非法/越界词项、重复文档 ID。
- `E_INDEX_INVALID`：非法 JSON、未知分片、未声明查询流版本、配置/兼容表/索引结构错误。
- `E_CONFIG_MISMATCH`：续跑时版本、manifest 或配置哈希不一致。
- `E_INTERACT_BUDGET_PENDING`：预算内无法完成最新参数，返回上一完整检索摘要。
- `E_EDIT_INVALID`：编辑命令结构或目标非法。

## 实际验收结果

2026-11-20 本地验收通过：

- 有效文档/查询：26 / 8。
- 最终格式：v2，档位 2，位置保留开启，147 个倒排块。
- 索引指纹：`cf376554f344598ec00f0c9278764bea4d75097e9b58f2cd79e140a9466a3557`。
- 逻辑倒排摘要：`7a68d967bf42a83924612497ff5b2c6a96e97c60e9ef5c1e269318b9243a106c`。
- 增量最终状态与空白全量重建：索引指纹、147 个块摘要、逻辑倒排、失败清单和 8 条查询结果全部一致，差异数 0。
- 档位 1/2 解码摘要一致；v2→v1→v2 三段逻辑摘要一致。
- 并发 1 与 4 的结果顺序及确定性耗时记录一致。
- 0ms 预算命令返回 `E_INTERACT_BUDGET_PENDING`，版本保持不变并返回上一完整摘要。
- 中断遗留 staging 与无指针孤儿快照在续跑时被清理，已完成版本不受影响。
- 配置哈希不匹配续跑返回 `E_CONFIG_MISMATCH`。
- 11 项 `unittest` 全部通过。

详细数据见 `reports/demo/consistency_report.json`、`reports/demo/consistency_report.md`；逐查询结果见 `reports/demo/incremental/retrieval_results.json` 与 `reports/demo/full/retrieval_results.json`。
