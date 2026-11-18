# 实际验证记录

验证日期：2026-11-18。目标运行环境为本地 Windows + Python 3.11+；本次执行宿主为 WSL2 中的 Ubuntu 22.04.5 LTS，使用 Python 3.11.0rc1、NumPy 2.4.6、pytest 9.2.0。没有把该宿主记录表述为 Windows 实机结果；Windows 版本可按 README 的命令直接复跑，代码无 POSIX 专用依赖。

## 环境与基线

- 验证宿主：`Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.35`
- Python：`3.11.0rc1 (CPython)`
- NumPy：`2.4.6`
- pytest：`9.2.0`
- venv：仓库根目录 `.venv`
- 运行依赖：`numpy<3,>=1.26`
- 运行网络：无；仅访问本地 `samples/`、`validation/` 和 `artifacts/`

完整环境、Python/NumPy 版本和样本摘要见 `artifacts/environment.json`。`pip freeze` 原文见 `artifacts/pip-freeze.txt`：

```text
iniconfig==2.3.1
numpy==2.4.6
packaging==26.3
pluggy==1.7.0
Pygments==2.21.0
pytest==9.2.0
slowprof==0.1.0
```

样本摘要：

```text
50ef47e46956315e7d4806ce9292a402828f584a2ac0da3d070a0a24a8d67a35  anomalies/depth_exceeded.ndjson
15c4b9d58a61f887712380ef47302bd8149f1794b67b55e8a0180be338421e8e  large.ndjson
0b8f594a06d9708c5601f5027c6592fe7634b992be196e9cbfb73af048b534a2  small.ndjson
7075564484db22cae7f8a1d3ff5da256111524506176f6bb50eb98559faad357  anomalies/snapshot_missing_version.json
643b9141811f45ef6aabf1198981830c5892a14b9b000f5f9cee5b110524666a  anomalies/zero_weight.ndjson
```

- 样本清单 SHA-256：`da4bd09749ad09801df4036b6353d7ad6f02cb051d042788bf15c982507374c6`
- 样本集摘要：`8c0762536f32a67581a8179b6e233207d93525a1e229f3e1da046fe6f3d83044`

## 自动化测试

命令：

```bash
.venv/bin/python -m pytest -q
```

结果：

```text
.....................................                                    [100%]
37 passed in 16.61s
```

覆盖项包括：字段强校验、权重为零、深度越界、时间戳回退、重复 JSON 键、区间过滤；同路径折叠、自耗时/累计耗时、递归重复帧去重；全量与流式路径字节一致；排名稳定规则、调用上下文、乱序输入和并发；定点误差、误差界决胜和 K 不足；快照版本门禁、恢复整体替换、只读归档；重放成功、首差异、样本摘要/口径/区间版本不匹配；CLI 三类注入与结构化日志。

## 三次会话结果

端到端命令：

```bash
.venv/bin/python -m slowprof demo --samples-dir samples --runtime-dir validation/runtime
```

| 会话 | 自动路径 | 区间内行数 | 总权重 | 总耗时 ms | 第一名 | canonical SHA-256 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| session-1-small-full | full | 12 | 24 | 2725 | http.call | `5fbc7d569b2d12e26b819291074263014a1f3116921e3aada3c0d7c8951c3f40` |
| session-2-large-streaming | streaming | 3131 | 64760 | 54486800 | ledger.write | `8727db2061a77358e0c36d223351799f1a310335147d61aee65968ac9e145a7e` |
| session-3-interval-changed | full | 9 | 17 | 2265 | http.call | `d08dc37155b93291cb64063fa27f2467a8b9442cd61825c56da19b63c3c15b93` |

第三次会话把区间起点从 `1700000000000` 改为 `1700000004000`，并把 `interval_version` 从 `obs-v1` 改为 `obs-narrow-v2`。三次快照均保存到 `validation/runtime/sessions/`，重放结果均为 `reproducible`。

## 双路径逐字节一致

对同一输入强制分别执行全量路径和流式路径，然后比较包含帧统计、排名、上下文和误差标注的 canonical JSON 字节：

| 数据集 | 行数 | 唯一栈 | 总权重 | 帧数 | 结果 | canonical SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| small | 12 | 8 | 24 | 11 | 一致 | `5fbc7d569b2d12e26b819291074263014a1f3116921e3aada3c0d7c8951c3f40` |
| large | 3131 | 125 | 64760 | 195 | 一致 | `8727db2061a77358e0c36d223351799f1a310335147d61aee65968ac9e145a7e` |

小样本自动选择 `full`，大样本自动选择 `streaming`；大样本为 3,131 行、125 个唯一栈、579,703 字节，其中 3,000 行集中在 4 个倾斜槽位（其中两条路径相同），另含 100 条唯一 worker 栈、20 条唯一数据库栈和一条深度 63 的深层栈。流式报告为 `streaming_no_path_strings: true`，固定深度工作缓冲为 `64 × 8 = 512` 字节。

## 快照恢复与重放

用第三次会话（`obs-narrow-v2`）恢复第一次会话快照（`obs-v1`）：

- 策略：`replace`，不增量合并。
- 不匹配字段：`interval.interval_version`、`interval.start_ms`、`sample.selected_samples_sha256`、`conclusion_sha256`。
- 变化帧：`api.handle`、`cache.get`、`db.query`、`sql.execute`。
- 排名变化：`billing.charge` 第 5 → 第 4；`redis.get` 未入榜 → 第 5；`cache.get` 第 4 → 未入榜。
- 旧快照归档：`session-1-small-full.readonly-*.json`，权限 `0444`，`archived_snapshot_read_only: true`。
- 一致恢复第三次会话：`restore`，无不匹配。
- 三次快照重放：均 `reproducible`。

## 异常注入

| 注入 | 业务码 | family | 行/字段 | CLI 退出码 | 结构化日志 |
| --- | --- | --- | --- | ---: | --- |
| zero-weight | `E_SAMPLE` | `E_SAMPLE` | 2 / `weight` | 2 | `validation/runtime-zero-weight/logs/slowprof.jsonl` |
| depth-exceeded | `E_DEPTH` | `E_SAMPLE` | 2 / `frames` | 3 | `validation/runtime-depth-exceeded/logs/slowprof.jsonl` |
| missing-version | `E_VERSION` | - | `metric_version` | 4 | `validation/runtime-missing-version/logs/slowprof.jsonl` |

注入仅改变选用的异常 NDJSON 或异常快照文件，不更换核心代码路径。所有错误日志均为 JSONL，包含 `code`、`family`、`line`、`field`、`details` 和 `injected`。

## 归档文件

- `artifacts/validation-report.json`：机器可读总报告，`all_passed: true`。
- `artifacts/demo-summary.json`：三次会话、双路径、恢复和重放摘要。
- `artifacts/pytest.txt`：测试原始输出。
- `artifacts/environment.json`：宿主、Python、NumPy、样本摘要。
- `artifacts/pip-freeze.txt`：冻结依赖。
- `validation/runtime*/sessions/`：快照和只读旧快照归档。
- `validation/runtime*/logs/`：结构化 JSONL 日志。

## 复跑

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python scripts/generate_samples.py --output-dir samples
.venv/bin/python scripts/run_validation.py
```

`scripts/run_validation.py` 会重新运行 pytest、正常三次会话、三类异常注入和环境记录，并写出 `artifacts/validation-report.json`。返回码为 0 表示全部验收项通过。
