# 偏好对齐奖励账本交付说明

本交付在本地进程内完成确定性合成、校验、特征哈希、定点打分、滑动汇总、分窗归并、双格式账本往返、增量失效、预算调度、快照恢复与故障注入。服务不监听端口，也不依赖外部服务或网络数据。

## 环境与复现

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m preference_ledger.cli verify --root .
python -m unittest discover -s tests -v
```

当前 WSL 镜像中的 `python3 -m venv` 缺少 `ensurepip`（提示需要 `apt install python3.11-venv`，且会话无免密 sudo）。实际使用已安装的 `virtualenv` 完成等价 PEP 405 环境创建：

```bash
python3 -m virtualenv --clear .venv
source .venv/bin/activate
pip install numpy pyyaml
```

实测版本：Python `3.11.0rc1`、numpy `2.4.6`、PyYAML `6.0.4`。

## 代码结构

- `preference_ledger/model.py`：记录、配置、问题码和枚举。
- `preference_ledger/loading.py`：JSONL、权重表、YAML 配置和编辑输入校验。
- `preference_ledger/util.py`：FNV-1a 哈希、规范指纹、子词切分、定点四舍五入和稳定排序键。
- `preference_ledger/ledger.py`：统一账本模型、g1 定宽行、g2 JSON、L1/L2 代际投影和旁路往返。
- `preference_ledger/engine.py`：特征哈希、成对打分、量化残差、滑动窗口、极值并列规则、分窗合成和诊断汇总。
- `preference_ledger/pipeline.py`：分阶段流水线、内容寻址缓存和逐字段比较投影。
- `preference_ledger/incremental.py`：编辑命令流、失效差异、复用清单和增量状态。
- `preference_ledger/snapshot.py`：临时目录写入、清单校验、指针原子替换和已提交快照恢复。
- `preference_ledger/flip.py`：量化/累加/阈值三类判定翻转归因。
- `preference_ledger/verify.py`：本地验收总入口，生成 `artifacts/verification_report.json`。
- `tests/test_ledger_system.py`：验收报告、全冲突标注者、边界窗口和第二组编辑序列测试。

## 样本与产物

- `samples/preferences.jsonl`：5 个初始会话、21 条多轮偏好记录，其中 `S006/ann_degenerate` 是全冲突退化样本。
- `samples/annotators.yaml`：整数万分比标注者权重。
- `samples/config.yaml`：16 维特征、32 个哈希桶、窗长 `[2,3,4]`、分窗边界 `[2,4,6]`。
- `samples/edits.yaml`：16 步编辑序列，覆盖替换、删除、会话删除/新增、权重调整、标注者增删、窗长/边界/量化/交换版本/代际切换。
- `samples/fault_injections.yaml`：5 个带注入编号的真实输入故障。
- `samples/ledgers/`：g1/g2 与 L1/L2 双版本双代账本样本及 g1 旁路元数据。
- `artifacts/verification_report.json`：机器可读验收结果。
- `artifacts/run_report.json`：4 线程全量运行报告。
- `artifacts/budget_state/state`：预算挂起后恢复的快照现场。
- `artifacts/crash_state/state`：真实 `SIGKILL` 后保留的已提交快照现场。

## 关键口径

### 特征与打分

- 子词规则：`sw-v1` 输出整词和三元子词；`sw-v2` 输出边界标记的二元/三元子词。
- 哈希：64 位 FNV-1a，桶下标为 `fnv1a_64(token) % hash_bucket_count`。
- 特征：桶向量按 `bucket % feature_dim` 投影，所有点积和汇总均按索引升序显式累加，不使用运行时随机状态。
- 权重：`weight_bp / 10000`。带权精确值为 `raw_score_diff * weight_bp * 10^places / 10000`。
- 量化档：`Q_COARSE=5`、`Q_BALANCED=2`、`Q_FINE=1`，单位为 `10^-places`；每条记录保存量化步长、网格整数、定点结果、分子/分母残差。

### 窗口与分窗

- 每个窗长生成尾轮不小于窗长的连续轮次窗口，按 `(session_id, end_turn, length)` 稳定排序。
- 窗口点数：`length * hash_bucket_count`。
- 最大/最小值并列时取最小轮次，并在 `tie_basis` 中记录候选轮次和决断依据。
- 边界窗口：精确等于阈值或差值绝对值不超过量化半步；输出高一位小数精度参考均值和差值。
- 分窗结论按窗口结束轮次、窗长升序，以窗口轮次跨度为权重在定点整数域合成。

### 账本往返

- g1 固定行宽为 77 字符，含版本、W/P 类型、作用域、会话、带符号千分比均值、判定码、跨度和轮次区间。
- g1 解析保留 `source_exchange=g1` 和 `missing_fields`。
- g1 升 g2 时千分比均值精确写为 `mean_permille / 1000`，并生成 `permille_exact:P3` 精度标签。
- g2 降 g1 使用 `ROUND_HALF_UP`，旁路保存截断残差、原精度标签和完整 `overwritten_entry`；凭旁路再升 g2 时逐字段还原。
- L1 JSON 为紧凑代际，L2 JSON 为富字段代际；两者解析后进入同一个 `LedgerBook/LedgerEntry` 结构。

### 缓存、预算与快照

- 桶向量、打分、窗口、分窗均有内容指纹键；上游键、标注者级权重指纹、窗参数、边界、量化档和交换版本按阶段进入键派生。
- 编辑后用旧/新缓存键映射生成失效清单，记录依据、涉及会话、是否跨会话和复用命中。
- 预算调度先按稳定顺序完成配额/预算判定，再在线程池中计算；最终结果统一排序。
- 预算不足的窗口进入 `pending` 并标记 `E_WINDOW_BUDGET_PENDING`；会话累计点数超配额时标记 `E_SESSION_QUOTA_EXCEEDED` 并中止该会话本批处理。
- 快照先写 `snapshots/txn-*` 临时目录，再整体 `rename` 为不可变 `snap-v*`，最后原子替换 `current.json` 指针。恢复只读取已提交且清单完整的快照，忽略未提交临时目录。

## 实测验收

`python -m preference_ledger.cli verify --root .` 当前结果为 `"ok": true`：

- 45 条窗口/分窗账本条目完成 g2 → g1 → 旁路 g2 逐字段相等往返，文件级往返也相等。
- L1/L2 核心字段归一相等；串行与 4 线程结果逐字段相等。
- 16 步编辑后，每一步的增量结果与相同最终输入的空白全量结果通过同一语义投影比较。
- 精确失效抽查通过：单轮替换只失效引用该轮的窗口；`ann_kim` 权重调整跨 4 个会话；窗长/边界编辑分别保留 26 个可复用窗口结论；量化档切换复用 17 个桶向量；增删无记录标注者为 0 失效。
- 128 点预算产生 32 个待办并命中 `E_WINDOW_BUDGET_PENDING`；恢复后续跑与全量一致；200 点会话配额命中 `E_SESSION_QUOTA_EXCEEDED`。
- 崩溃子进程真实返回 `-9`（`SIGKILL`），重启后从版本 1 已提交快照恢复，再跑结果与全量一致。
- 5 个注入项全部命中：`E_VERSION_UNKNOWN`、`E_FIELD_TRUNCATED`、`E_ANNOTATOR_WEIGHT_RANGE`、`E_NON_FINITE`、`E_WINDOW_MONOTONIC`；21 条健康样本仍可继续处理。
- 翻转归因的三个确定性用例分别输出 `score_quantization`、`weighted_accumulation`、`threshold_comparison`。

`python -m unittest discover -s tests -v`：6 个测试全部通过。
