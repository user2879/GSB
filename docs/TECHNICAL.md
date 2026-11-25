# ReconFlow 技术说明

ReconFlow 是一个本地、确定性的训练样本流复核包。它把领域声明、滚动 NDJSON 批次、配比/清洗配置作为唯一输入，产出加权采样、近重复去重、配比归一三阶段结论，并提供增量缓存、权威重算、TCP 分片调度、SQLite 账本恢复和 Tkinter 诊断台。

## 1. 快速运行

```bash
python3 --version                 # 需要 >= 3.10
python3 -m pip install -r requirements.txt
export PYTHONPATH=src

python3 -m reconflow demo --root artifacts/demo
python3 -m reconflow ui --root artifacts/demo
python3 -m unittest discover -s tests -v
```

也可以直接运行 `./scripts/verify.sh`。包本身提供 `pyproject.toml` 与兼容旧 pip 的 `setup.py`，可执行 `python3 -m pip install .` 后使用 `reconflow` 命令。

## 2. 目录与模块

- `src/reconflow/specs.py`：JSON/NDJSON 加载、强校验、`E_SPEC` 行号/字段定位。
- `src/reconflow/synthetic.py`：固定种子的 BLAKE2b 计数随机流、文档/批次生成、内容摘要。
- `src/reconflow/fixedpoint.py`：万分之一定点、稳定余数分配、Int64 饱和规则。
- `src/reconflow/pipeline.py`：加权采样、精确有理相似度去重、配比归一、误差归因、候选裁决、`E_DIVERGE` 定位。
- `src/reconflow/incremental.py`：编辑、失效传播、仅失效分片重算、材料化快速视图、权威比对。
- `src/reconflow/rpc.py` / `worker_server.py`：127.0.0.1 TCP NDJSON 握手、超时、回执字段校验和子进程托管。
- `src/reconflow/scheduler.py`：全局/领域双预算、排队 monotonic 计时、远程分片聚合。
- `src/reconflow/ledger.py` / `repository.py`：SQLite 格式版本、批次结局、失效轨迹、读取耗时、游标、阶段项、快照、SIGKILL 恢复和 `E_REPLAY`。
- `src/reconflow/ui.py`：Tkinter/ttk 诊断界面与无 Tk 时的稳定文本投影。
- `src/reconflow/demo.py`：一键本地验收和错误码自检。

## 3. 规范与错误模型

所有加载先完整校验到新对象，成功后才允许进入后续流程；不会把半条记录写入工作缓存。

支持范围：

- 种子、文档数、批次大小：非负/正整数且可写入 Int64。
- 采样权重：整数 `0..10000`，单位为万分之一，至少一个为正。
- 相似度阈值：整数 `5000..9999`，单位为 basis point。
- 分片大小：整数 `1..1024`。
- `missing_bitmap`：0/1 数组，长度必须等于 `term_counts`；`length` 必须等于计数和。

首个非法记录抛出结构化 `E_SPEC`，包含 `path`、`line`、`field`、`value`。NDJSON 行号即物理行号；JSON 数组通过 `JSONDecoder.raw_decode` 扫描对象区间并定位字段行。

错误码：

- `E_SPEC`：规范校验失败。
- `E_DIVERGE`：快速视图与权威重算不一致。
- `E_REPLAY`：账本重放不一致或仍有 unknown/pending 批次。
- `E_PORT`：回环端口占用/无法绑定。
- `E_TIMEOUT`：握手、hello 或回执读取超时。
- `E_FIELD`：握手或回执缺少/错配字段。

## 4. 确定性样本与摘要

随机数不依赖 NumPy 全局 RNG 或 Python 进程哈希随机化，而是对规范版本、领域种子、文档序号、用途 salt 和计数器计算 BLAKE2b keyed-style 摘要并取 64 位整数。

每篇文档包含：

- `doc_id`、`domain_id`、`doc_seq`、`batch_seq`
- `term_counts`
- `length = sum(term_counts)`
- `missing_bitmap`
- `content_digest = sha256(canonical_bytes(...))`

摘要求绑定领域、逻辑批序号、文档序号和词项计数分布。文档序列只由领域声明决定；物理 TCP/读取“批次切分”可以任意重组传输块，但不改写逻辑 `batch_seq` 与摘要。领域处理和输出一律按 `domain_id` 排序，因此领域文件加载顺序不影响结果。

规范 JSON 示例由 `reconflow generate` 写入：

- `specs/domains.json`
- `config/config.json`
- `config/candidates.json`
- `data/batches.ndjson`

## 5. 三阶段整数定点契约

### 5.1 加权采样

设总采样点 `B`，领域相对权重 `w_d`，权重和 `W`。先取：

```text
q_d0 = floor(saturate_i64(B * w_d) / W)
```

未分配点数按余数 `(B*w_d) mod W` 降序、再按 `domain_id` 升序补发。配额超过可用文档时截断为可用文档数，未用点不跨领域转移，并标记 `quota_saturated=true`。

### 5.2 近重复去重

词项多重集相似度使用精确整数有理数：

```text
I = sum(min(c1_i, c2_i))
U = sum(max(c1_i, c2_i))
near_duplicate iff 10000 * I >= threshold_bp * U
```

不使用浮点相似度。贪心代表选择按 `doc_seq`、`doc_id` 稳定排序；同一输入在本地、TCP 子进程、任意并行度下都聚合为相同分片摘要。

### 5.3 配比归一

实际配比和目标配比都用同一稳定余数法分配到合计 `10000‱`。`deviation_bp = actual_bp - target_bp`。

所有除法均为非负整数的 floor（Python `//`），余数补发规则显式且稳定。所有乘法/加法在持久化前经 `saturate_i64` 钳到 signed Int64 上限；演示数据远低于溢出阈值，但规则写入配置 `saturation_rule`。

### 5.4 规范化字节

所有比较使用：

```python
json.dumps(value, sort_keys=True, separators=(",", ":"),
           ensure_ascii=False, allow_nan=False) + "\n"
```

结论摘要是对排除 `conclusion_digest` 自身后的规范字节再做 SHA-256。

## 6. 快速视图、权威重算与失效传播

编辑入口：

```bash
reconflow edit-weight alpha 4000
reconflow edit-threshold 8800
reconflow edit-shard-size 8
reconflow edit-doc alpha:d00003 --term-counts '[...]' --missing-bitmap '[...]'
```

每次编辑：

1. `config_version += 1` 并重新强校验配置。
2. 重新计算轻量加权采样和分片边界。
3. 比较旧/新 `members_digest`；阈值编辑额外用缓存中的精确 `intersection/union` 判断哪些分片的布尔判定会越界翻转。
4. 生成 `config -> domain -> shard -> dedup -> ratio` 的稳定步骤和依据。
5. 只把 `affected_shards` 交给本地或 TCP 算力。
6. 未受影响分片直接复用旧结论对象；随后只重算便宜的配比归一。
7. 比较配比项规范字节，得到真正变化的 `affected_ratio_domains`。
8. 原子替换 `cache/fast_view.json` 与 `cache/fast_conclusion.canonical.json`。

界面查询只执行一次 `read_bytes()` + JSON 解析，不调用全量管线；读取耗时来自 `time.monotonic_ns()`。`reconflow view --record` 或 API `record_fast_read` 才把耗时写入 SQLite。Tk 的普通“只读刷新”不写账本。

权威路径 `reconflow verify-view` 从原始 JSON/NDJSON 重新执行完整管线并写字节文件。若有差异，`first_divergence` 按 `weighted_sampling -> shards -> ratio_normalization -> error_attribution`、分片/领域 ID 序返回首个 `stage/domain_id/shard_id/field`，CLI 以 `E_DIVERGE` 退出。

## 7. TCP 子进程与双预算调度

Worker 由 `start_worker_subprocess` 以 `python -m reconflow.worker_server` 启动，绑定 `127.0.0.1:0` 后输出 ready 行。协议为真实 TCP 上的 NDJSON：

1. Server 发 `hello {"protocol":1}`。
2. Client 发 `handshake`。
3. Server 回 `handshake_ack`。
4. 每个 domain-shard 任务一行；server 回带 `result` 与 `digest` 的 receipt。

所有 socket 操作设置真实 `settimeout()`。hello/握手/回执超时统一报 `E_TIMEOUT` 并带 `phase`；占用端口报 `E_PORT`；缺少 `type/task_id/ok/result/digest` 或摘要不符报 `E_FIELD`。

调度器为每个任务计算全局点数和 `{domain: points}`，在条件变量后等待：

```text
active_global + task_global <= global_sampling_point_budget
active_domain[d] + task_domain[d] <= domain_sampling_point_quota[d]
```

获取预算前后都在同一锁内更新 active/peak；等待开始、结束的 monotonic 毫秒和真实等待时长写入 `wait_records`。结束后校验 `peak_global_points <= budget`、`peak_domain_points[d] <= quota` 且 `invariant_violations=[]`。

## 8. SQLite 账本、杀死恢复与重放

账本路径：`ledger/reconflow.sqlite`，`PRAGMA user_version=1`。

主要表：

- `batch_plan`：预期批次和状态。
- `batch_outcomes`：按 `(batch_id, config_version)` 保留的批次结局修订。
- `invalidations`：失效集合、步骤和摘要。
- `ui_reads`：monotonic 读取耗时、字节数和摘要。
- `cursors`：批次/结论/UI 游标。
- `stage_items`：每版本、批次来源的阶段规范项。
- `recovery_events`：unknown 标记。
- `snapshots`：SQLite 备份快照文件的 SHA-256 与元数据。

正常批次结局、计划状态和游标在同一事务提交。崩溃夹具在执行 DML 后、commit 前写 marker 并睡眠；父进程确认 marker 后发送 `SIGKILL`（验收实测子进程返回码 `-9/137`）。未提交事务由 SQLite 回滚；重启后只能看到已提交记录，所有没有 outcome 的计划批次被标记为 `unknown` 并要求重跑。

文档编辑会改变原始批次摘要。持久化当前视图时，仓储层比较已有最新批次结局与新原文；若不同，就在登记失效轨迹和阶段项的同一事务中追加新的 `batch_outcomes` 修订。旧修订和 SQLite 快照仍保留供审计。

`reconflow replay` 先检查 unknown/pending，再逐批次核对 outcome，随后按账本 item 顺序重放 weighted/dedup/ratio/error 项。首个差异以 `E_REPLAY` 给出 `batch_id`、`stage`、`domain_id`、`shard_id`、`field` 和双方摘要。

快照使用 SQLite Online Backup 写到 `snapshots/*.sqlite`，同时写 JSON 清单和文件 SHA-256。

## 9. 误差归因与不可区分裁决

内部误差单位为 micro-basis-point（`1‱ = 1000 u`），按阶段顺序保守求和：

1. 采样整数点粒度：`ceil(1000 * 10000 / sampling_budget_points)`。
2. 去重边界归属：统计阈值边界带内涉及的文档数，按领域采样数换算为配比影响上界。
3. 归一累积：最大余数法在每个坐标上的最坏 `1‱ = 1000 u`。

各阶段上界换算为千分比贡献 `share_permille`，稳定分配到合计 `1000`；主导判定阈值显式声明为 `600/1000`。若最高贡献达到 600，则 `dominant_above_threshold=true`。

两个候选方案先计算相对当前实际配比的最大绝对偏差分数；当分数差 `abs(score_a-score_b) <= total_bound_micro_bp` 时，认为误差界内不可区分，按 `candidate_id` 字典序选择，并设置：

```json
"indistinguishable_under_error": true,
"error_affected": true
```

若分数差超过误差界，则选择定点偏差更低者，平局仍按 `candidate_id` 字典序。

## 10. Tkinter 诊断界面

`reconflow ui --root artifacts/demo` 打开四个标签页：

- 逐阶段结论：采样配额、分片 docs/kept/dropped、配比偏差和摘要。
- 失效集合/传播依据：完整稳定 JSON 步骤。
- 误差归因/裁决：阶段贡献、主导阶段、候选分数和选择原因。
- 未知批次/快照/读取：unknown 标记、快照 SHA、最近读取耗时。

Treeview 和 JSON 均按 domain/shard/id 排序。“只读刷新”只调用 `collect_diagnostic_state()` 的 SELECT/文件读取，不写账本。无图形环境或 Tk 库缺失时，`reconflow.ui.headless_diagnostic_text()` 提供同样顺序的文本投影；demo 会写出 `diagnostic_view.txt`。
