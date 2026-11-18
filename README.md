# 长对话摘要与改写工作台

本工作区实现一个完全本地、确定性的中英对话增量摘要工作台。输入为固定种子的会话事件 JSONL、受限词表、摘要配置和编辑命令流；输出逐轮摘要、来源映射、失败清单、复用/重算范围、阶段耗时统计和可恢复快照。

## 环境与快速开始

```bash
python3 --version                 # 3.10+
python3 -c 'import numpy; print(numpy.__version__)'

# 固定种子重新生成样本（当前 samples/ 已提交）
python3 -m dialogwb generate-samples --out samples

# 基线运行 + 三批编辑，写入结果、报告与快照
python3 -m dialogwb run \
  --events samples/events.jsonl \
  --lexicon samples/lexicon.json \
  --config samples/config.json \
  --edits samples/edits.json \
  --out artifacts/demo \
  --snapshot artifacts/demo-snapshots \
  --fresh

# 中断后续跑
python3 -m dialogwb run ... --resume

# 查看当前完整快照清单
python3 -m dialogwb status --snapshot artifacts/demo-snapshots

# 完整验收矩阵
python3 -m dialogwb validate --samples samples --out artifacts/validation

# 回归测试
python3 -m unittest discover -s tests -v
```

依赖仅需 Python 3.10+ 与 NumPy 1.24+，不使用 GPU、容器、网络账号或外部服务。`requirements.txt` 与 `pyproject.toml` 均已提供。

## 输入格式

`events.jsonl` 每行包含 `session_id`、`turn`、`speaker`、`text`、`timestamp`。加载器逐行隔离非法 JSON，并校验会话 ID、轮序、空话语和时间戳；合法话语按词表与子词规则切分，每个词项保留原话语中的 `[start, end)` 字符区间。

词表文件包含 `tokens`、`rules.case_sensitive`、`rules.punctuation`、`rules.subwords` 与 `version`。英文词项要求词边界，中文按最长词项匹配；子词规则优先。词表外片段输出 `E_UNKNOWN_TOKEN`。

业务失败码：

- `E_BAD_JSON`：非法 JSON 行。
- `E_EMPTY_UTTERANCE`：空话语。
- `E_UNKNOWN_TOKEN`：词表外词项，消息携带字符区间。
- `E_TURN_GAP`：轮序重复、倒退或空洞；该行进入失败清单，后续行继续处理。
- `E_BAD_TIMESTAMP`、`E_EMPTY_SESSION`、`E_BAD_EVENT`：补充的加载诊断。
- `E_STAGE_EXCEPTION`：抽取、合并或改写阶段被隔离的运行时异常。

配置支持压缩窗口、摘要最大长度、候选上限、要点基础/衰减/词项权重、模板权重、去重阈值、快照间隔与并发数。编辑批支持 `replace`、`delete`、`add`、`set_compression_window` 和嵌套 `config_patch`。删除会压缩后续轮号，新增会在指定位置插入并顺延后续轮号，从而保持编辑后输入仍为严格连续轮序。

## 核心确定性规则

- 会话按 `session_id`、轮次按 `turn` 稳定排序；线程任务的稳定键为 `(turn_rank, session_id)`。
- 窗口为当前轮与前 `compression_window - 1` 轮；被移出窗口的轮次写入 `evicted_turns`。
- 话语权重由词项计数、词项权重、句子奖励和位置衰减计算；候选按权重降序、来源轮次升序、片段起始位置升序决断。NumPy `lexsort` 用于稳定数组排序。
- 近重复使用词项集合 Jaccard 相似度。代表条目取最高权重；同权重取较小摘要条目序号。来源条目序号写入合并记录。
- 模板由摘要事实的 SHA-256 指纹映射到累计权重区间，保证确定性加权选择。超长摘要优先在句界截断，并记录截断位置、原长度、保留长度和丢弃字符数。
- 逐轮输出包含摘要条目、来源映射、摘要长度、合并/截断信息和阶段复用标记。

## 增量缓存与失效

三个阶段的缓存键分别由以下内容派生：

1. 抽取：窗口内话语内容指纹、窗口配置、要点权重、词表指纹。
2. 合并：抽取产物指纹、上一轮摘要状态指纹、去重阈值、词表指纹。
3. 改写：合并条目指纹、模板/最大长度配置、词表指纹。

编辑后从头按稳定轮序比较缓存键。窗口不覆盖被修改话语的抽取产物直接复用；一旦上游或累计状态变化，对应合并/改写及后续演进重算。结果中记录每个会话的 `reused_rounds`、`recomputed_rounds` 以及三个阶段各自的复用/重算轮次。

## 快照与恢复

每次阶段完成时，完整不可变状态先写入私有临时目录，状态使用确定性 gzip 写入；随后目录原子改名到 `versions/vNNNNNNNN`，最后原子替换 `CURRENT` 指针。因此进程在任意写入时刻被强杀，读者只能看到最后一个完整版本。

清单记录快照版本、配置哈希、词表指纹、游标、阶段、摘要状态指纹、各阶段产物版本和运行状态。恢复时校验基础/当前配置哈希、词表指纹、基础输入指纹、编辑流指纹和版本一致性；不一致返回 `E_CONFIG_MISMATCH` 并拒绝恢复，`CURRENT` 指针不变。

- `SIGINT`/`SIGTERM`：完成当前轮并提交快照，写出结果与报告，以退出码 `0` 结束。
- 强制终止：不写半成品；从最后完整 `CURRENT` 恢复，并在结果中输出 `resumed_from`、游标和阶段。
- 阶段中途恢复：抽取/合并/改写阶段游标与待处理产物均包含在快照状态中。

`stage_timing` 使用确定性的 `work_units`/`logical_ms`，因此串行与并发、全量与复用运行的该字段可逐字段比较；真实墙钟单独放在 `runtime_metrics.observed_wall_clock_ms`，不作为一致性判定字段。

## 输出

`result.json` 主要字段：

- `sessions.*.rounds`：逐轮抽取、合并、改写、摘要条目、来源映射和复用标记。
- `sessions.*.final_summary`、`final_source_map`：最终摘要与来源。
- `summary_lengths`：逐会话逐轮摘要长度与条目数。
- `counts.merge`、`counts.truncation`：合并和截断计数。
- `failures`：稳定排序的失败清单。
- `stage_timing`、`runtime_metrics`：逻辑阶段工作量与实际观测耗时分列。
- `recovery`、`resumed_from`、`config_hash`、`vocab_fingerprint`、`snapshot_version`。

`report.json` 汇总上述验收诊断字段。核心一致性视图由 `dialogwb.engine.core_view` 提供，排除快照版本、墙钟和复用元数据。

## 实际验证结果

当前样本共 36 行事件，包含 7 个会话（其中 3 个故障样本），基线 32 个有效摘要轮次。完整验收结果见 `artifacts/validation/validation_report.json`：

- 必需业务码全部命中：`E_BAD_JSON`、`E_EMPTY_UTTERANCE`、`E_UNKNOWN_TOKEN`、`E_TURN_GAP`。
- 基线快照版本 28；三批编辑最终增量快照版本 115，全量重建版本 31。
- 基线、替换批次、删除/新增/窗口批次、英文替换/权重批次均与对应全量重建逐字段一致。
- 最终摘要：33 轮、合并 73 次、截断 16 次、失败 4 条；增量与全量 `first_difference = null`。
- 并发 1 与并发 4：摘要、来源、失败顺序与 `stage_timing` 完全一致。
- `SIGTERM`：退出码 0，在快照版本 4 中断，恢复后与不中断结果一致。
- 强杀：退出码 137，从完整快照版本 5 恢复，恢复后与不中断结果一致。
- 配置哈希不一致：返回 `E_CONFIG_MISMATCH`，快照指针未变化。
- 注入合并异常：异常被记录为 `E_STAGE_EXCEPTION`，后续轮次不再引用失败话语。

`all_checks_passed = true`。单元/集成测试共 12 项全部通过。

## 源码结构

```text
dialogwb/
  cli.py          CLI 与信号/退出码
  config.py       配置校验、补丁与配置哈希
  edits.py        编辑批应用与轮号归一化
  engine.py       三阶段演进、缓存、并发、结果与核心视图
  lexicon.py      受限词表与子词切分、字符区间
  loader.py       JSONL 加载、业务校验与失败清单
  samplegen.py    固定种子样本生成
  snapshot.py     阶段原子快照与指针发布
  validation.py   完整验收矩阵
  workbench.py    基线/编辑批编排、恢复校验与报告
tests/            12 个单元与集成测试
samples/          固定种子输入
artifacts/        实际运行与验收产物
```
