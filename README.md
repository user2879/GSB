# 确定性流式语音指令解析应用

本工作区实现了一个仅依赖 Python 与 NumPy 的本地样本应用。它把固定种子生成的分片声学打分表映射为会话全局帧轴，在对数域构建候选格图，以贪心或有限窗口维特比解码，并在响应预算内返回最近一次已完成结果。交互编辑后的最终语义结果与从零全量解码逐字段一致。

## 1. 交付内容

- `streamparse/`：加载、校验、格图、重叠归属、解码、槽位、指纹缓存、并发、快照、预算交互与诊断报告实现。
- `scripts/generate_samples.py`：固定种子生成词表、打分表、会话流、配置、注入配置与编辑样本。
- `scripts/run_acceptance.py`：运行全量/增量等价、预算、并发、注入、快照验收并写出报告。
- `scripts/run_demo.py`：演示策略切换、参数修改、分片追加/替换、会话增删。
- `tests/`：14 个 `unittest` 测试。
- `sample_data/`：已生成的确定性样本，种子 `20261120`。
- `artifacts/`：实际验收报告、初始诊断报告、增量诊断报告与原子快照。

## 2. 实际环境与复现

本次实际运行环境：

| 项目 | 值 |
|---|---|
| 操作系统 | Linux / WSL2（`Linux-6.18.33.2-microsoft-standard-WSL2-x86_64`） |
| Python | `3.11.0rc1`，解释器 `/usr/bin/python3` |
| NumPy | `2.4.6` |
| `python` 别名 | 当前环境无该别名，使用 `python3` |
| pip | `26.3` |

代码使用 Python 3.10+ 语法和 `pathlib`/`os.replace`，Windows 下可将下列命令中的 `python3` 换成 `python` 或 `py -3.11`。本工作区实际验证在 WSL2 中完成，未在原生 Windows 进程内重复执行。

```bash
python3 -m pip install -r requirements.txt
python3 --version
python3 -c "import numpy; print(numpy.__version__)"
```

生成确定性样本：

```bash
python3 scripts/generate_samples.py --out sample_data --seed 20261120
```

运行测试与完整验收：

```bash
python3 -m unittest discover -s tests -v
python3 scripts/run_acceptance.py
python3 scripts/run_demo.py --sample-dir sample_data --workspace-root .
```

## 3. 输入格式

### 3.1 `vocab.json`

词表条目包含 token、词性、槽位起点标记、槽位名和子词切分；同时提供开始标记、平滑词、子词规则和二元回退语言模型。

```json
{
  "version": "zh-instruction-20261120",
  "start_token": "<s>",
  "smoothing_token": "<sil>",
  "subword_rules": {"strategy": "restricted-longest-match", "token_separator": "", "subword_separator": "|"},
  "entries": [
    {"token": "打开", "pos": "VERB", "slot_start": true, "slot": "command", "subwords": ["打", "开"]},
    {"token": "空调", "pos": "NOUN", "slot_start": true, "slot": "device", "subwords": ["空", "调"]}
  ],
  "bigrams": [{"prev": "打开", "token": "空调", "score": 1.8}]
}
```

### 3.2 `scores.jsonl`

每行是一个分片局部帧的候选词发射分数向量。`session_id` 用于多会话样本；为兼容只带分片序号的打分表，也支持省略该字段并使用全局分片行。

```json
{"session_id":"S01","shard_seq":0,"frame_in_shard":0,"scores":[-100,-100,2.4,0.4]}
```

分数向量长度必须等于词表大小且全部有限。所有分数低于 `candidate_missing_below` 的帧视为缺候选，使用 `smoothing_score` 补入 `smoothing_token` 并记录位置。

### 3.3 `sessions.jsonl`

```json
{"session_id":"S01","shard_seq":0,"start_frame":0,"end_frame":4,"arrived":true}
```

`arrived` 可省略，默认 `true`。会话内分片序号必须从 0 开始、严格递增且无空洞；分片起止帧必须与打分表局部帧数一致。帧坐标是全局帧轴的权威来源，`overlap_frames` 是重叠裁决策略、输入约束和失效审计参数。实际重叠帧数与配置值都会写入 `overlap_audit`，因此替换或编辑后出现不一致也可审计。

### 3.4 `decode_config.json`

必需字段包括 `overlap_frames`、`min_stable_frames`、`lm_weight`、`slot_min_frames`、`slot_max_frames`、`response_budget_ms`、`thread_pool_concurrency`、`snapshot_interval_shards`。可选字段包括 `strategy`、`lookahead_frames`、`lm_backoff_score`、`smoothing_score`、`candidate_missing_below`、`stable_margin`、`snapshot_root`、`report_path` 与仅用于预算测试的 `test_compute_delay_ms`。

### 3.5 `injection_config.json`

支持 `non_finite_score`、`frame_count_mismatch`、`overlap_not_less_than_shard`、`zero_lookahead` 与 `invalid_slot_bounds`。每项带唯一 `id`；报告中的 `injection_report.items` 记录命中次数、实际阶段、配置业务码和观察业务码。

## 4. 交互命令

CLI 接收逐行 JSON。每次编辑命令触达后最多等待 `response_budget_ms`；超时返回 `E_RESPONSE_BUDGET_PENDING` 和最近一次完成摘要，后台继续计算。

```bash
python3 -m streamparse.cli --sample-dir sample_data --workspace-root . --interactive
```

常用命令：

```json
{"cmd":"set_strategy","session_id":"S01","strategy":"greedy"}
{"cmd":"set_param","session_id":"S01","key":"lookahead_frames","value":2}
{"cmd":"set_param","session_id":"S01","key":"overlap_frames","value":1}
{"cmd":"append_shard","session_id":"S01","shard_seq":3,"start_frame":8,"end_frame":11,"score_file":"extras/append_scores.jsonl"}
{"cmd":"replace_shard","session_id":"S01","shard_seq":1,"start_frame":3,"end_frame":6,"score_file":"extras/replace_scores.jsonl"}
{"cmd":"add_session","session_id":"S05","shards":[{"shard_seq":0,"start_frame":0,"end_frame":4,"score_file":"path/to/S05_shard0.jsonl"}]}
{"cmd":"delete_session","session_id":"S03"}
{"cmd":"wait"}
{"cmd":"status"}
{"cmd":"report"}
```

`scores` 可以是 `[{"frame_in_shard":0,"scores":[...]}, ...]`，也可以是按局部帧顺序排列的向量列表。`wait` 是显式同步命令；普通编辑命令不会为等待最新结果而突破预算。

## 5. 确定性语义

### 5.1 全局帧轴、重叠裁决与格图

1. 按 `(session_id, shard_seq)` 排序分片，使用记录中的 `start_frame/end_frame` 映射到全局帧轴。
2. 同一全局帧若被多个分片覆盖，权威归属键为 `(shard_seq, revision)`，后到分片优先；同序号替换使用递增 `revision`。
3. 每个被覆盖的前片帧均写入 `ownership_conflicts`，包含获胜/失败分片、局部帧号、修订号和全局帧号。
4. 每个权威帧把有限发射分数与词表候选组成格图卡片；二元转移分数在解码时以 `emission + lm_weight * bigram_backoff` 在对数域叠加。
5. 无有效候选的帧补 `smoothing_token`，并在 `smoothed_frames` 与逐帧决策的 `smoothed` 字段标注。

### 5.2 贪心与前瞻

- 贪心：按当前前词选择 `emission + LM` 最高候选，立即向右推进。
- 前瞻：从稳定边界开始，对每个位置在 `[position, position + lookahead_frames]` 窗口内做维特比，回溯取得该位置 token，再固定该 token 推进。
- 两种策略共享同一所有权表、同一全局帧轴、同一格图；策略指纹不参与格图指纹，因此切换策略不会使格图缓存失效。

### 5.3 稳定前缀与可变尾部

稳定前缀使用确定性裕量证书，而不是依赖编辑历史：

1. 先按贪心路径计算每帧最优与次优分数差 `margin`；
2. 从首帧开始统计 `margin >= stable_margin` 且非平滑的连续帧；
3. 连续帧数至少达到 `min_stable_frames` 才认证前缀；
4. 每帧需要其后的连续证据，因此最新 `min_stable_frames - 1` 帧始终保留在可变尾部。

该定义与策略、前瞻窗口无关，所以切换贪心/前瞻不会重写稳定前缀。前瞻只从稳定边界之后运行。若替换分片改变稳定前缀覆盖帧，应用回溯到最早受影响分片区间，失效稳定前缀和尾部并写入 `affected_intervals`。

### 5.4 指纹与增量失效

每个会话报告五类指纹：

- `ownership_fp`：权威归属表、分片坐标和修订号。
- `grid_fp`：词表、归属、发射分数、平滑参数，不包含策略与重叠策略。
- `stable_fp`：稳定前缀内容、词表、最小稳定帧数和裕量阈值。
- `tail_fp`：格图、稳定前缀、策略、前瞻、重叠策略和尾部逐帧结果。
- `strategy_fp`：当前策略参数指纹。

缓存审计同时给出 `reuse`、`partial_invalidate`、`reuse_with_extension` 或 `invalidate`，并统计复用/失效帧数。典型行为：

- 追加分片：只失效新增帧与重叠区，旧稳定前缀可复用并向后扩展。
- 替换分片：从新旧分片最早起点到会话下游失效。
- 修改策略/前瞻/重叠：格图与稳定前缀复用，尾部解码失效。
- 修改 LM/平滑/候选阈值：格图及其下游失效。

### 5.5 响应预算

编辑命令先提交状态快照，再把计算提交到后台串行计算器。前台只等待到预算截止点；若未完成：

- 返回 `status = pending` 和业务码 `E_RESPONSE_BUDGET_PENDING`；
- 摘要来自最近一次已完成版本，含稳定前缀文本、可变尾部帧区间、已填充槽位与受影响区间；
- 后台任务继续运行；版本过期结果在提交报告/快照前被识别并丢弃。

### 5.6 并发、原子快照与异常隔离

- 分片格图任务和会话尾部解码任务由 `thread_pool_concurrency` 控制的线程池执行。
- 任务稳定排序键为 `(session_id, shard_seq, start_frame, revision)`；最终会话、帧和失败清单顺序与单线程一致。
- 单分片异常只产生一条带会话 ID、分片序号、阶段和业务码的失败记录，其余分片和会话继续。
- 每个阶段先写入快照根目录下的临时目录，写文件哈希与 `manifest.json`，再整体替换 `current/`。提交前后均检查全局版本；变化时返回 `E_SNAPSHOT_STALE`，不替换已有快照。
- 清单包含 `global_version`、`config_hash`、`completed_frame_cursor`、阶段名、文件哈希和提交状态。

## 6. 诊断报告

每次计算完成写出 `artifacts/diagnostic_report.json`。报告包含：

- 环境、全局版本、配置哈希、线程池设置；
- 按会话排序的解析摘要；
- 稳定前缀、可变尾部、最终 token/文本、槽位帧区间；
- 逐帧策略输出与贪心参考输出；
- 所有帧归属和重叠冲突记录；
- 平滑帧、策略一致前缀、指纹、缓存动作和受影响区间；
- 排序后的失败清单与注入命中报告；
- 最终阶段快照摘要。

运行 `scripts/run_acceptance.py` 还生成 `artifacts/diagnostic_report_incremental.json` 和 `artifacts/acceptance_report.json`。

## 7. 实际验证结果

测试命令 `python3 -m unittest discover -s tests -v`：14/14 通过。覆盖输入异常隔离、会话序号空洞、槽位下界校验、重叠归属审计、平滑位置、策略共享格图、编辑序列逐步全量等价、缓存失效、预算 pending、线程数 1/4 语义一致、快照原子提交与陈旧版本拒绝。

验收命令 `python3 scripts/run_acceptance.py`：8/8 检查通过，本次耗时约 6.0 秒。关键实测：

- 4 个注入项全部命中，分别映射 `E_INPUT_INVALID`、`E_INPUT_INVALID`、`E_INPUT_INVALID`、`E_DECODE_PARAM_INVALID`。
- 非有限数注入只使目标分片失败；`S-INJ-SHARD` 仍以其余分片完成解析并标记 `partial`。
- 重叠帧 `3,4` 由分片 1 覆盖分片 0；帧 `5,6` 由分片 2 覆盖分片 1。
- 线程数 1 与 4 的 `semantic_payload`、失败顺序、归属记录和指纹完全一致。
- 10 次编辑（策略切换、前瞻/重叠修改、追加、替换、会话增删、LM 修改）每次均与同状态全量解码逐字段一致。
- 20 ms 预算下实际返回耗时约 19.6 ms，返回 `pending` 和旧版本摘要；随后 `wait` 完成并返回 `ok`。
- 快照清单包含版本 0、配置哈希、8 个会话的完成帧游标与 `decode.json` 等阶段产物。

## 8. 代码结构

```text
streamparse/
  loader.py       输入读取、逐条校验、JSONL 容错
  models.py       词表、配置、分片、格图、结果数据模型
  grid.py         分片校验、全局帧归属、格图与指纹
  decoder.py      贪心、窗口维特比、稳定前缀
  slots.py        槽位帧区间填充
  engine.py       并发全量语义、缓存审计、诊断报告
  app.py          交互状态、预算截止、后台推进、编辑命令
  snapshot.py     临时目录 + 整体替换的原子快照
  cli.py          JSONL 命令 CLI
  samplegen.py    固定种子样本生成
scripts/          生成、演示、验收脚本
tests/            自动验收测试
```

## 9. 边界说明

- 全部输入为本地程序生成的合成分数，不使用音频、真实声学模型或联网服务。
- 稳定前缀采用确定性裕量证书；这是流式场景下的可复现认证语义，不等同于无限未来上下文下的理论全局最优前缀。
- `overlap_frames` 修改只影响重叠策略与尾部缓存失效，不重排输入记录中已经给出的全局帧坐标；实际与配置重叠差异通过 `overlap_audit` 显式呈现，以避免交互修参导致历史帧坐标漂移。
