# Dialogue RAG 技术说明

## 运行

```bash
pip install numpy
python -m dialogue_rag.sample_data --out samples
python -m dialogue_rag.cli run \
  --config samples/config.json \
  --commands samples/commands.json
```

默认输出：

- 诊断报告：`artifacts/diagnostic_report.json`
- 持久化快照：`artifacts/state/manifest.json` 与 `artifacts/state/commits/v*`

可通过 `--output`、`--state-dir`、`--workers` 覆盖；`--no-resume` 忽略已有清单重放命令流。

## 领域流程

1. `model.py` 加载并校验多会话、轮次、证据库、通道、召回声明、定点参数和预算。
2. `recall.py` 计算 `raw_score * channel_weight`，按加权得分降序、通道标识升序选择权威通道，并合并同轮重复证据。
3. `topics.py` 使用确定性词元/Jaccard 相似度切分话题；相似度与阈值恰等时按 `topic.equal_rule`（`merge` 默认或 `split`）决断，并写入 `equal_boundaries`。
4. `scoring.py` 按来源轮次序号升序、证据 ID 字典序升序遍历引用链。每个“来源轮次 × 证据”的实值因子为：

   ```text
   contribution = evidence_weight
                * citation_count
                * distance_decay ** (target_seq - source_seq)
                * (cross_topic_penalty if evidence/source crosses topic else 1)
   ```

   定点域使用 `scale = 2 ** fixed_width`，当前允许 `F8/F16/F32`；`quant_gear=G0/G1/G2/G3` 对应额外的 `0/4/16/64` LSB 舍入网格。乘法后采用非负数半入舍入。
5. `diagnostics.py` 用精确有理数作为高位宽参考，输出召回打分、去重合并、话题归属、置信度合成四级误差，并可比较两份报告，判定 `quantization_rounding`、`dedup_or_recall_changed` 或 `topic_attribution_changed`。

## 指纹与精确失效

缓存键由以下指纹派生：

- 轮次内容：会话、轮次 ID、序号、查询；
- 证据内容与证据权重；
- 召回声明、通道权重、上限；
- 话题阈值与恰等规则；
- 距离衰减、跨话题惩罚、归一化和结论阈值；
- 定点位宽与量化档位。

依赖传播关系：

- 查询变化使该轮召回/去重键失效，并沿 `source_turn -> target_turn` 链检查下游评分；
- 话题规则变化重算切分与归属，只有成员指纹变化的话题进入 `recompute_topics`；
- 证据权重变化只使该证据的定点贡献键失效，其他证据贡献可复用，但相关轮次会重新求和与合成结论；
- 证据归属变化通过 `evidence -> topic set` 交叉引用使引用该证据的评分轮次失效；
- 位宽/档位进入召回、贡献和置信度键，因此内部整数与结论会真实重算。

报告中的 `invalidation.cache_decisions` 给出每轮每阶段的 `reused/recomputed/pending`、缓存键和命中依据；`invalidation.evidence_decisions` 给出证据级 `topic_attribution/forensic_weighting` 状态。

## 预算与并发

一个取证点数等于引用链中的一个“来源轮次 × 证据”条目。预算不足时按稳定排序键 `(session_id, seq, turn_id)` 保留后续待办，并写入 `E_BUDGET_PENDING`；下一批优先处理这些键。

`state.workers > 1` 时使用 `ThreadPoolExecutor`。工作线程只读取状态和既有指纹缓存，评分结果回到主线程后统一写缓存；任务携带相同稳定排序键，结果最终按键排序，因此与串行逐字段一致。

## 快照与恢复

`SnapshotStore` 每次提交创建 `tmp/txn-N/`，写入 `state.json`、`history.json`、`pending.json`、`failures.json` 和 `stages/*.json`，计算文件 SHA-256 与阶段版本；未变化阶段硬链接到新事务目录，然后整体 `os.replace` 到 `commits/vN`，最后原子替换根 `manifest.json`。

清单包含：

- `global_version`
- `config_hash`
- `evidence_fingerprint`
- `command_cursor`
- `stage_versions` 与 `stage_changed`
- 文件哈希和快照目录

恢复时校验清单和所有文件哈希。清单缺失、JSON 损坏、快照目录缺失或哈希不一致时返回 `E_SNAPSHOT_REBUILD`；校验通过时返回 `I_SNAPSHOT_RESUME` 并从 `command_cursor` 继续。SIGINT/SIGTERM 处理器只设置停止事件，执行器等待在飞轮次完成、不再调度新轮次，随后提交含待办的快照。

## 测试

```bash
python3 -m pytest tests/test_dialogue_rag.py -q
```

测试覆盖校验隔离、权威通道、恰等阈值、定点变化、量化翻转归因、轮次/证据精确失效、预算、并发一致性、8 组确定性随机命令序列的重启一致性、快照恢复/损坏重建和信号处理。
