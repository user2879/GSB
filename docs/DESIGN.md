# 设计与可复算规范

## 1. 模块边界

| 模块 | 职责 |
| --- | --- |
| `batch_audit/dataset.py` | 确定性样本与两层 MLP 参数构造、NPZ 持久化 |
| `batch_audit/feed.py` | JSON 强校验、规范排序、同槽去重、槽位分组 |
| `batch_audit/forward.py` | 固定算术顺序的 NumPy 前向推理与输出摘要 |
| `batch_audit/ledger.py` | SQLite schema、槽位事务提交、内容摘要、恢复 |
| `batch_audit/engine.py` | 快照版本对账、调度、报告、强制终止注入 |
| `batch_audit/audit.py` | 只读账本与原始投递的独立重放审计 |
| `batch_audit/gui.py` | Tkinter 本地工作台 |
| `batch_audit/cli.py` | 命令行入口与结构化错误输出 |

审计器不调用 `feed.py`、`forward.py` 或 `engine.py` 的业务函数。它从账本读取原始样本/参数数组，在 `audit.py` 内独立完成排序、分组、编号推导和推理。

## 2. 确定性样本与模型

`generate_assets(seed, sample_count, input_dim, hidden_dim, output_dim)`：

- 用 `seed & 0xFFFFFFFF` 得到 32 位种子。
- 输入矩阵使用独立 `RandomState` 流生成 `uniform(-1, 1)`，形状为 `(sample_count, input_dim)`。
- 每层权重/偏置使用各自派生的 `RandomState` 流。
- 模型为两层 `tanh` 隐藏层：`input_dim -> hidden_dim -> hidden_dim -> output_dim`。
- 全部数组持久化为小端 `float64`、C 顺序。

参数矩阵与输入矩阵不依赖投递文件、槽位交错或条目书写顺序。

## 3. 排序、去重、分组

1. 结构字段先做强校验。错误以 `E_FEED` 整体拒绝。成功条目必须引用本地确定性样本；准入失败条目不进入推理，可携带外部标识。
2. 全部条目按 `(arrival_seq, source_index)` 排序。`source_index` 只用于校验前的确定性定位；最终数据要求 `arrival_seq` 唯一。
3. 准入失败条目不参与分组，但保留其槽位和原因。
4. 每个槽位内按 `sample_id` 收集成功投递：
   - 保留 `(arrival_seq, source_index)` 最小者；
   - 其余写入覆盖清单，记录被保留条目的到达序号；
   - 同一成功样本若出现在不同槽位，形成歧义，整体 `E_FEED`。
5. 每个槽位保留者再按 `(arrival_seq, sample_id, source_index)` 排序，名次从 1 开始。
6. 无成功成员的槽位登记 `E_EMPTY`，不影响其他槽位。

文件条目书写顺序不会改变规范分组结果；`source_index` 只是原始路径元数据。

## 4. 规范字节与摘要

### 4.1 canonical JSON

```text
UTF-8, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
```

### 4.2 数组记录

数组先转为小端 `float64`、C 顺序，负零归一为正零。每条数组记录为：

```text
canonical_json({"dtype":"<f8","name":NAME,"order":"C","shape":[D0,...]})
+ b"\n"
+ little_endian_float64_C_bytes
```

单数组摘要：

```text
sha256(b"ARRAY-DIGEST-V1\n" + array_record + b"\n")
```

### 4.3 输入摘要

样本标识按字符串排序，输入矩阵按同样的样本顺序重排：

```text
sha256(
  b"INPUT-DIGEST-V1\n" +
  canonical_json({"sample_ids": sorted_ids, "shape": [rows, cols]}) + b"\n" +
  array_record("inputs", sorted_rows) + b"\n"
)
```

### 4.4 参数摘要

固定顺序：`W1, b1, W2, b2, W3, b3`。

```text
sha256(
  b"MODEL-DIGEST-V1\n" +
  concat(array_record(name, tensor) + b"\n")
)
```

### 4.5 前向推理与输出摘要

线性层不使用 BLAS；按输入索引从小到大依次执行 NumPy 乘加：

```text
result = bias
for i in range(input_dim):
    result = result + vector[i] * weight[i]
```

两层隐藏激活和输出激活均为 `tanh`。输出先四舍五入到 12 位小数并归一负零。

样本输出摘要：

```text
sha256(
  b"SAMPLE-OUTPUT-DIGEST-V1\n" +
  canonical_json({"sample_id": ID, "dtype":"<f8", "shape":[out_dim]}) + b"\n" +
  output_float64_bytes + b"\n"
)
```

批次输出摘要按样本标识排序绑定成员，不依赖组内名次：

```text
sha256(
  b"BATCH-OUTPUT-DIGEST-V1\n" +
  canonical_json({"members": [[sample_id, sample_output_digest], ...sorted]}) + b"\n"
)
```

### 4.6 归属编号

哈希载荷：

```json
{
  "batch_slot": 0,
  "model_param_digest": "<64-hex>",
  "numbering_version": "BA-V1",
  "rank": 1,
  "sample_id": "S0000"
}
```

```text
full_digest = sha256(b"BATCH-ASSIGNMENT-V1\n" + canonical_json(payload) + b"\n")
label = "<version>:S<slot>:R<rank:06d>:<full_digest[:20]>"
```

编号输入不包含输出摘要、文件顺序、Python 字典迭代顺序或进程状态。

### 4.7 投递摘要

条目规范化后按 `arrival_seq` 排序，仅包含：

```text
sample_id, arrival_seq, batch_slot, admitted, reject_reason
```

```text
sha256(
  b"FEED-DIGEST-V1\n" +
  canonical_json({"schema": schema, "entries": normalized_entries}) + b"\n"
)
```

### 4.8 账本内容摘要

账本内容摘要覆盖：

- 所有 generation 的逻辑字段；
- 数组名称、dtype、shape、单数组摘要；
- 所有槽位状态、成员快照、覆盖、失败、输出摘要、游标；
- 所有归属记录；
- 所有执行尝试及其状态。

按 generation、slot、rank、attempt 的固定顺序 canonical JSON 后计算 SHA-256。`ledger_header.format_version` 为整数格式版本，当前为 `1`。审计器会独立重算该摘要。

## 5. SQLite 账本与槽位事务

关键表：

- `ledger_header`：整数格式版本与内容摘要。
- `generations`：版本、五元指纹、配置、样本标识、进度、替换来源。
- `arrays`：账本内嵌的输入和模型数组，供审计器离线复算。
- `slots`：每槽状态、成员、覆盖、失败、输出摘要、游标。
- `assignments`：编号、名次、样本、样本输出摘要、批次输出摘要。
- `attempts`：运行尝试；强杀后保持 `running`。

每个槽位完成时，在一个 `BEGIN/COMMIT` 事务内：

1. 删除该槽旧归属，写入当前编号集合；
2. 更新槽位成员、覆盖、失败、输出摘要与槽位游标；
3. 更新 generation 进度游标和已完成槽位数；
4. 更新 attempt 最后提交槽位；
5. 重算并写入账本内容摘要。

SQLite 使用 WAL 与 `synchronous=FULL`。进程在事务中退出时，该事务不会成为已提交状态。

## 6. 强制终止与恢复状态机

- 新执行在 `attempts` 中写入 `running`。
- 每提交一个槽位后更新快照；快照不包含退出报告。
- `os._exit(99)` 测试注入绕开 `finally`、异常处理和报告写出。
- 重启时发现 `running` attempt：
  - attempt 改为 `interrupted`；
  - 未提交槽位改为 `unknown`，错误码 `E_UNKNOWN_REQUIRES_REDO`；
  - 记录未知槽位后重新执行。
- 已提交槽位不会重算，已提交编号保持不变。
- 只有全部槽位处于 `committed` 或 `empty` 后才写最终 `report.json`。

因此，强制终止时系统不提供、也不声称提供“终止时退出报告”。

## 7. 版本对账与陈旧编号防护

版本指纹：

```text
feed_digest, input_digest, param_digest, numbering_version, config_digest
```

每次运行比较：

1. 账本最新 generation 的指纹；
2. 磁盘 `snapshot.json` 的版本与指纹；
3. 当前输入文件与参数计算出的指纹。

只有三者一致才续跑。以下任一情况整体创建新版本，而不是合并槽位：

- 投递、输入、参数、编号口径或配置变化；
- 快照缺失；
- 快照版本与账本版本不一致；
- 账本存在但对应 generation 不可续跑。

新版本号：

```text
max(ledger_max_generation_id, snapshot_generation) + 1
```

旧 generation 标记为 `superseded`，其归属和输出不出现在当前报告/审计中。报告稳定输出旧版本受影响样本数与已产生的槽位。

## 8. 独立审计与不一致定位

审计器：

1. 以 SQLite `mode=ro` 打开账本；
2. 验证整数格式版本、账本内容摘要与每个数组摘要；
3. 从账本读取样本标识、输入矩阵、模型数组和 generation 配置；
4. 只读本地原始投递 JSON，独立完成校验、排序、分组与去重；
5. 独立计算编号、样本输出摘要和批次输出摘要；
6. 按槽位、名次逐项比较状态、成员、名次、编号、摘要、覆盖、失败与错误码；
7. 输出可重复的 `recomputed_digest`。

比较顺序稳定。业务不一致先于账本内容摘要检查报告，因此篡改某条输出摘要时，`first_mismatch` 能定位到该样本、槽位与字段：

```json
{
  "code": "E_AUDIT",
  "first_mismatch": {
    "sample_id": "S0002",
    "batch_slot": 2,
    "field": "sample_output_digest",
    "expected": "...",
    "actual": "tampered"
  }
}
```

## 9. 强制终止后的审计边界

可审计槽位：账本状态为 `committed` 或 `empty`。

不可审计槽位：

- `pending`：尚未执行；
- `running`：执行中未提交；
- `unknown`：中断恢复时标记需重做；
- 缺失槽位行。

不可审计槽位不会被推测，也不会从内存或退出报告补齐。审计结果为 `AUDIT_PARTIAL`，并逐槽列出原因 `NOT_COMMITTED_BEFORE_TERMINATION_OR_NOT_YET_EXECUTED` 或 `NO_COMMITTED_SLOT_ROW`。

## 10. 边界说明

- 同槽重复成功投递是允许的重投；跨槽成功重复是归属歧义并整体拒绝。
- 推理避免 BLAS 形状差异并使用 12 位小数归一，降低跨环境末位差异风险；逐字节复算仍以相同 Python/NumPy 兼容环境为准。
- 账本摘要用于独立复算与完整性检查，不是密钥签名，不能抵御同时重写业务数据和内容摘要的恶意攻击者。
