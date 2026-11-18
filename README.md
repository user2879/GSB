# slowprof：慢请求确定性调用栈剖析

`slowprof` 面向本地 Windows 环境，输入为 NDJSON 请求聚合样本，输出可持久化、可恢复、可重放的慢请求热点剖析会话。运行时只访问本地文件和目录，不依赖网络、数据库或外部服务。

## 交付内容

- `src/slowprof/`：可安装 Python 包，支持 Python 3.11+。
- `samples/`：确定性小样本、大规模倾斜/深层样本和三份异常夹具。
- `scripts/generate_samples.py`：样本再生成脚本。
- `examples/run_demo.py`：三次剖析会话、双路径校验、快照恢复和重放示例。
- `tests/`：严格校验、双路径字节一致、排名、定点误差、恢复、重放和 CLI 异常测试。
- `docs/VALIDATION.md`、`artifacts/`：实际运行环境、依赖冻结和验收记录。

## Windows 环境准备

在仓库根目录执行（PowerShell 示例）：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts\generate_samples.py --output-dir samples
.\.venv\Scripts\python.exe -m pip freeze > artifacts\pip-freeze.txt
.\.venv\Scripts\python.exe examples\run_demo.py --samples-dir samples --runtime-dir runtime
.\.venv\Scripts\python.exe -m pytest
```

包只使用 `pathlib`、`os.replace`、`os.chmod` 和标准文件 API，不依赖 POSIX 专用调用。`runtime\sessions` 和 `runtime\logs` 由程序自动创建，进程只需对 `runtime` 有读写权限。

本次实际验证在 WSL/Linux 宿主完成，宿主 Python 为 3.11.0rc1；目标 Windows 路径和代码已按可移植方式实现。真实宿主记录见 `artifacts/environment.json`，没有把 Linux 结果冒充为 Windows 实机结果。

## NDJSON 样本契约

每行一个 JSON 对象：

```json
{"request_id":"req-1","timestamp_ms":1700000000000,"duration_ms":120,"service_id":"checkout","frames":["api.handle","db.query","sql.execute"],"weight":2}
```

强校验规则：

- 必填字段为 `request_id`、`timestamp_ms`、`duration_ms`、`service_id`、`frames`、`weight`。
- `request_id`、`service_id` 为非空字符串。
- `timestamp_ms` 为整数，整条文件流内单调不降。
- `duration_ms`、`weight` 为正整数；布尔值不被当作整数。
- `frames` 为非空列表，深度不超过 `max_depth`。
- 每个帧名严格符合 `svc.method`，即一个点分隔的两段标识符。
- 空行、坏 JSON、非 UTF-8、非法类型都拒绝整次加载，不返回部分结果。
- 区间之外的行同样执行结构和时间戳校验；只有区间内行进入聚合。

观测区间采用闭区间 `[start_ms, end_ms]`。任一样本错误都带行号和字段。帧深度错误使用业务码 `E_DEPTH`，同时标记 `family=E_SAMPLE`，表示它属于“整条样本流拒绝”的 `E_SAMPLE` 契约。

## 耗时口径

对一条样本，记其耗时 `T`、权重 `w`；同一有序帧路径的权重在折叠表中合并。重复帧名在一个栈中只计一次参与，避免递归栈重复累计。

```text
self_time(f)        = Σ_{栈顶/叶帧为 f 的样本} w × T
cumulative_time(f)  = Σ_{包含 f 的样本}         w × T
participant_weight  = Σ_{包含 f 的样本}         w
occurrence_count    = 包含 f 的样本条数
```

因此非叶帧的 `self_time_ms` 为 0；叶帧的 `self_time_ms == cumulative_time_ms`。所有帧的 `self_time_ms` 之和等于会话的总加权耗时。折叠表只按路径合并权重；不同样本即使路径相同、耗时不同，帧指标仍按每条样本的实际耗时累计。结果按帧名排序，与输入行顺序无关。

调用上下文按栈中唯一直接邻接边聚合：

```text
edge_inclusive_time(caller, callee) = Σ_{包含该邻接边的样本} w × T
```

同一条栈中重复出现的同一邻接边只计一次。上下文同时输出 `inclusive_time_ms` 和参与权重，并按耗时降序、权重降序、帧名字典序稳定排列。

## 双聚合路径与阈值

默认阈值在 `ProfilePolicy` 中：

- `full_unique_stack_threshold = 64`
- `full_total_weight_threshold = 4096`
- 仅当 `unique_stack_count < 64` 且 `selected_total_weight < 4096` 时走全量折叠路径，否则走流式路径。

全量路径保存路径折叠表，适合唯一栈较少的小规模数据。流式路径使用 NumPy 固定深度数组保存当前栈 ID，跨行不保留任何完整栈路径字符串；帧统计、上下文边和固定深度工作数组大小与唯一栈数弱相关。为了选择路径，先做一次只保留 128 位 BLAKE2b 栈摘要计数的强校验/元数据遍历，再做一次单遍聚合遍历；报告会明确写出 `streaming_no_path_strings: true`。

报告包含 `path`、阈值、唯一栈数、总权重、决策规则、确定性峰值内存估算和估算组成。由于估算只使用结构和计数、不读取进程 RSS，同一输入重复运行得到相同数值。两条路径的帧统计、排名、上下文和误差标注以规范化 JSON 字节比较；测试同时覆盖小样本和大样本的强制双路径一致性。

## 排名与定点误差

排名固定为：

```text
self_time_ms 降序 -> cumulative_time_ms 降序 -> frame 字典序升序
```

`K` 为正整数；不足 K 个帧时输出实际数量。排名使用无界整数精确值，不使用量化后的定点数，因此前 K 名集合必然是真实精确比较器的前 K 名。报告仍给出边界证据和该判定依据。

占比按会话参数 `scale` 量化：

```text
fixed_ratio(f) = round_half_up(self_time_ms(f) × scale / total_time_ms)
```

单帧量化误差精确表示为有理数；单帧误差上界为 `1/(2×scale)`，两帧合成误差界为 `1/scale`（即一个定点单位）。两个已报告热点帧的定点差不超过 1 个单位时，报告标记 `quantization_order_error_bound_dominated` 并列出 `error_band_ties`；排序仍使用显式的精确稳定决胜规则，不因量化模糊而抖动。前 K 合计占比的量化误差上界为 `K/(2×scale)`，并同时给出实际有理误差。

## 快照、恢复与重放

快照包含：

- 快照 schema 版本和口径版本 `metric_version`。
- 闭区间参数及 `interval_version`。
- 样本文件 SHA-256、区间内规范化样本 SHA-256、行数、唯一栈数和策略。
- 聚合路径、阈值依据和内存估算。
- 帧折叠结果、热点排名、直接调用者/被调用者、口径说明和全部误差标注。全量路径另存精确栈折叠表；流式路径按“无完整路径字符串驻留”契约省略该表，并写入 `stack_folds_omitted_reason`，帧级折叠结果仍完整持久化。
- 结论规范化字节的 `conclusion_sha256`。

恢复时先读取并校验版本，再比较样本摘要、区间参数/版本、口径版本、策略与结论摘要。任何不一致都不做逐帧增量合并，而是以当前会话的完整状态整体替换。结果稳定列出 `changed_frames` 和 `rank_moves`，并把旧快照复制为带只读权限的 `.readonly-*.json` 归档。只读位在 Windows 会映射为文件只读属性。

重放严格采用快照记录的闭区间和聚合路径，重新加载同一文件，逐项比较帧统计、排名和误差标注。第一处差异输出帧名、字段、快照期望值和重算值，并以 `E_REPLAY` 报告。样本 SHA、区间版本或口径版本不匹配时，报告具体不匹配项。

## CLI 与异常注入

Python 集成入口：

```python
from slowprof import (
    Interval, ProfilePolicy, aggregate_file, build_ranking, canonical_core_bytes,
    load_samples, profile_file, replay_snapshot, restore_snapshot, save_snapshot,
)

policy = ProfilePolicy(top_k=5)
interval = Interval(1_700_000_000_000, 1_700_000_020_000, "obs-v1")
samples, metadata = load_samples("samples/small.ndjson", policy, interval)
session = profile_file("samples/small.ndjson", policy, interval)
save_snapshot(session, "runtime/sessions/demo.json")
restored = restore_snapshot("runtime/sessions/demo.json", session)
replayed = replay_snapshot("runtime/sessions/demo.json", "samples/small.ndjson")
```

`examples/api_usage.py` 单独展示加载、双路径聚合、排名、快照、恢复和重放；`examples/run_demo.py` 展示三次会话与三类注入。`restore_snapshot` 返回 `RestoreOutcome`，其 `active_session` 是整体替换后的完整活动状态。

```powershell
# 三次会话完整演示
.\.venv\Scripts\python.exe -m slowprof demo --samples-dir samples --runtime-dir runtime

# 三类业务码异常，退出码分别为 2、3、4
.\.venv\Scripts\python.exe -m slowprof demo --samples-dir samples --runtime-dir runtime --inject zero-weight
.\.venv\Scripts\python.exe -m slowprof demo --samples-dir samples --runtime-dir runtime --inject depth-exceeded
.\.venv\Scripts\python.exe -m slowprof demo --samples-dir samples --runtime-dir runtime --inject missing-version

# 单独剖析、双路径校验、重放和恢复
.\.venv\Scripts\python.exe -m slowprof profile --samples samples\large.ndjson --start-ms 1710000000000 --end-ms 1710001000000 --snapshot runtime\sessions\large.json
.\.venv\Scripts\python.exe -m slowprof parity --samples samples\large.ndjson --start-ms 1710000000000 --end-ms 1710001000000
.\.venv\Scripts\python.exe -m slowprof replay --snapshot runtime\sessions\large.json --samples samples\large.ndjson
```

`--inject` 只切换异常输入文件或异常快照；核心校验、聚合、日志和错误映射逻辑不变。错误同时打印到 stdout，并追加到 `runtime\logs\slowprof.jsonl`，记录 `code`、`family`、`line`、`field`、`details` 和注入类型。

业务码：

| 码 | 含义 | CLI 退出码 |
| --- | --- | --- |
| `E_SAMPLE` | 样本字段/结构/时间戳/权重等非法 | 2 |
| `E_DEPTH` | 栈深度越界，同时属于 `E_SAMPLE` 契约 | 3 |
| `E_VERSION` | 快照口径或 schema 版本缺失/不支持 | 4 |
| `E_REPLAY` | 重放不可复现或结果不一致 | 5 |
| `E_PARAM` | 参数非法 | 6 |
| `E_IO` | 本地文件或目录错误 | 7 |

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

测试覆盖：严格字段校验、零权重、深度越界、时间戳回退、区间过滤、同路径权重折叠、自耗时/累计耗时、重复帧去重、调用上下文、双路径字节一致、输入乱序、并发稳定性、定点误差与误差界决胜、K 不足、快照版本门禁、一致/不一致恢复、只读归档、重放成功、首差异定位、摘要/口径/区间版本不匹配和 CLI 结构化异常日志。

实际运行结果、样本 SHA-256、依赖冻结和 canonical 摘要见 `docs/VALIDATION.md` 与 `artifacts/`。
