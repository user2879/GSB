# 消息投递语义格式互操作与版本兼容实验台

这是面向本地 Windows/Python 3.10+ 的确定性实验台。它不发送网络消息，只处理本地 JSON/JSONL 输入：把 v1、v2、v3 三种历史投递语义记录归一为同一内部模型，推导每条消息的 `at_least_once`、`at_most_once`、`exactly_once` 结论，并保证原始行乱序、消费者组处理乱序时结论字节稳定。

主流程输出可由独立实现重新归一、推导和统计并做逐字节校验；SQLite 账本按主题批次原子提交，支持强制终止恢复、参数版本演进、陈旧结论隔离、快照整体替换和五类业务错误码。

## 环境准备

```powershell
py -3.10 -c "import sys, sqlite3; print(sys.version); print(sqlite3.sqlite_version)"
py -3.10 -m pip install -r requirements.txt
py -3.10 -m mwlab selfcheck --config samples/session.json
```

依赖仅 `numpy==1.26.4`；其余全部使用标准库。Windows 自带的 Python 安装器需包含 `tkinter`。当前交付环境是 Linux 容器，`sqlite3` 与 NumPy 正常，但 `tkinter` 因缺少 `libtk8.6.so` 只能给出 `pass_with_warnings`；核心 CLI 不依赖 GUI，在 Windows 上可用 `--require-tkinter` 做严格检查。

## 快速演示

所有命令都在仓库根目录执行。

```powershell
python -m mwlab init --config samples/session.json
python -m mwlab run --config samples/session.json --records samples/records.jsonl --expected samples/expected_summary.json
python -m mwlab report --artifact out/artifact.json --out out/report.json
python -m mwlab verify --config samples/session.json --records samples/records.jsonl --model-digest out/model_digest.txt --conclusions out/conclusions.jsonl --group-conclusions out/group_conclusions.jsonl --conflicts out/conflicts.jsonl --stats out/stats.json
python -m mwlab snapshot save --config samples/session.json --name baseline
python -m mwlab snapshot restore --config samples/session.json --name baseline
python -m mwlab recover --ledger state/ledger.sqlite3 --records samples/records.jsonl
```

完整验收流程可一键运行：

```powershell
python scripts/demo.py
# 或
powershell -ExecutionPolicy Bypass -File scripts/demo.ps1
```

演示覆盖归一、推导、定点统计、冲突归因、独立逐字节校验、快照恢复、强制终止恢复、参数演进和全部异常注入。

## 样本与输入格式

- `samples/session.json`：4 个主题、4 个消费者组；组格式分别为 v1/v2/v3/v3。
- `samples/session_evolved.json`：改变 `orders` 确认策略并新增 `shadow` 组，用于参数版本单调递增和陈旧视图替换。
- `samples/records.jsonl`：34 条混合格式记录，包含 9 条 v1、9 条 v2、16 条 v3。
- `samples/expected_summary.json`：预期结论、冲突计数、格式计数和内容摘要。
- `samples/records_bad_range.jsonl`、`samples/records_bad_version.jsonl`：`E_FORMAT` 样本。

每行是一个带信封的记录。v1 载荷仍只有必需的三字段；消费者组由声明提供并在信封中给出上下文。

```json
{"format_version":"v1","consumer_group":"legacy","record":{"message_id":"orders:1","topic":"orders","acknowledged":true}}
{"format_version":"v2","consumer_group":"billing","record":{"message_id":"orders:1","topic":"orders","acknowledged":true,"delivery_time_ms":1700000200001,"retry_count":0,"consumer_group":"billing"}}
{"format_version":"v3","consumer_group":"analytics","record":{"message_id":"orders:1","topic":"orders","acknowledged":true,"delivery_time_ms":1700000200001,"retry_count":2,"consumer_group":"analytics","idempotency_key":"idem-orders-1-a","delivery_sequence_range":{"start":0,"end":2},"semantic_verdict":"exactly_once"}}
```

强校验包括：未知版本、必需/多余字段、组声明版本不匹配、非整数毫秒、负重投次数、倒置区间、非法幂等键或语义标记。错误详情包含 `line` 与 `field`；归一结果保留 `source_format`、`source_group`、`source_line`。

## 语义规则

对每个 `(topic, message, consumer_group)` 细胞：

1. `attempts = max(record_count, max(retry_count)+1)`。
2. v3 投递区间按 `[0, attempts-1]` 裁剪并合并；只有存在区间数据时才计算“未覆盖区间”。
3. 确认策略要求确认、幂等键和完整区间时，三者都满足才得到 `exactly_once`。
4. 未达到精确一次，但存在确认、重投或未覆盖区间时，得到 `at_least_once`。
5. 其余单次、未确认、无区间缺口的情况得到 `at_most_once`。

全局消息结论按组细胞汇总：全部精确一次才精确一次；全部至多一次才至多一次；否则至少一次。所有结论按主题升序、消息序号升序、消息标识升序输出。

冲突账本包含 `ACK_WITH_UNCOVERED_RANGE`、`DUP_IDEMPOTENCY_ACROSS_GROUPS`、`SEMANTIC_MARKER_MISMATCH` 和策略触发的 `RETRY_LIMIT_EXCEEDED`，按消息序号、冲突类型字典序稳定输出。

## 定点统计与误差界

比例为 `floor(numerator * 10^scale_power / denominator)`。样本使用 `scale_power=6`，非负比例等价于向零截断。单值真实量化误差严格小于 `1/scale_factor`；两个定点比例比较的合成误差界严格小于 `2/scale_factor`。若两个主题定点差 `<= 2` 个定点单位，则标记 `error_affected=true`，并使用主题字典序作为稳定决胜规则。

当前样本中 `orders=4/16=250000` 与 `payments=3/12=250000` 的定点差为 0，因此两者均标记受误差影响，并按字典序稳定排序。

## 账本与快照

- SQLite 表保存归一记录、全局/组结论、冲突、按主题+组定点统计、误差界、主题排名、进度游标和视图差异。
- 每完成一个主题，在该主题事务内同时写入记录、结论、冲突、统计、批次摘要和游标。
- 参数指纹变化时创建新的单调递增参数版本，旧的批次、结论、统计整体标为陈旧，不参与当前查询和汇总。
- 快照保存 `snapshot.json` 和完整 SQLite 副本，含内容摘要与格式版本集合。覆盖同名快照前，旧快照复制到只读备份目录。
- 恢复时校验快照摘要并比较当前账本；格式集合或内容摘要不一致时仍执行整体复制替换，绝不增量合并。旧账本同样保留为只读备份。

## 强制终止与重放

```powershell
python -m mwlab run --config samples/session.json --records samples/records.jsonl --ledger state/crash.sqlite3 --out out-crash --crash-after-topic 1
python -m mwlab recover --ledger state/crash.sqlite3 --records samples/records.jsonl
```

第一条命令以退出码 91 强制终止，只留下第一个完整主题批次。恢复输出 `status=incomplete`、未提交主题、`unknown_uncommitted_topic` 消息、`exit_report_available=false`，不会伪造终止时退出报告。重新运行完整流程后再恢复，会从账本记录序列按已存参数版本重推结论；若发现差异则报告 `E_REPLAY` 及首个消息、字段和两侧取值。

## 业务错误码

| 错误码 | 退出码 | 触发方式 |
|---|---:|---|
| `E_CONFIG` | 2 | 声明 JSON 不符合约束 |
| `E_FORMAT` | 3 | 记录格式损坏，或 `run --inject-format` |
| `E_DUPKEY` | 4 | 跨消费者组幂等键重复被提升为致命错误，或 `run --inject-dupkey` |
| `E_SNAPSHOT` | 5 | 快照摘要不符，或 `snapshot restore --inject-digest` |
| `E_DIVERGE` | 6 | 独立校验与主流程不一致，或 `verify --inject-divergence` |
| `E_REPLAY` | 7 | 账本记录重推与已提交结论不一致 |
| `E_STATE` | 8 | 不完整执行、快照或视图状态约束失败 |
| `E_IO` | 9 | 文件读取/写入失败 |

错误同时写入结构化 JSONL 日志，默认 `logs/structured.jsonl`，可用 `--log-file` 覆盖。

## 测试

```powershell
python -m unittest discover -s tests -t . -v
```

测试覆盖格式强校验、行/组乱序字节稳定、语义结论、冲突排序、定点误差、独立校验、参数版本与陈旧隔离、快照整体恢复、强制终止恢复及 `E_FORMAT`/`E_DUPKEY`/`E_SNAPSHOT`/`E_DIVERGE`/`E_REPLAY`。

更详细设计见 `docs/DESIGN.md`，实际验证数据见 `docs/VALIDATION.md`。

