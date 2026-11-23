# 技术设计

## 1. 确定性模型与摘要

每个后端按声明构造四组参数：

- `W1: model_dim x hidden_dim`
- `b1: hidden_dim`
- `W2: hidden_dim x model_dim`
- `b2: model_dim`

随机源为显式 SplitMix64；浮点样本从高 24 位映射到 `[-1, 1]`，矩阵按行优先展开。`fp32-v1` 将权重存储轮换到 float32 精度，`int8-sym-v1` 和 `int4-sym-v1` 使用对称逐矩阵 scale 量化。前向：

```text
hidden = max(x @ W1 + b1, 0)
logits = hidden @ W2 + b2
output_i64 = rint(logits * 1_000_000)
```

输出摘要绑定后端标识、参数摘要、样本标识、输入摘要、输出整数与缩放口径。参数摘要绑定后端标识、模型/隐藏维度、种子、量化口径及其版本、参数版本、权重摘要。

热更新只允许修改参数相关字段；每次更新 `version += 1`，摘要与快照后端集合摘要随之变化。账本保存实际采纳回执的参数摘要及版本。

## 2. 稳定分流与批次

```text
bucket = (first_u64(SHA256(canonical_json({
  "algorithm": "gray-merge-lab-route-v1",
  "rule_id": ...,
  "client_key": ...
}))) % 10000) + 1
```

桶值域为 1..10000，是要求的 0..10000 域内的精确万分比切片：

```text
destination = "canary" if bucket <= canary_weight_bp else "stable"
```

批次规范化步骤：

1. 样本按 `(sample_seq, sample_id, client_key)` 排序。
2. 按 `(rule_id, destination, client_key)` 分组。
3. 每组按规则声明 `batch_size` 切块。
4. 批次标识由规则、去向、客户端键、样本标识、序号与批大小规范摘要生成。

选择客户端同质批，是为了使历史 h2 的单值 `client_bucket` 对整批语义一致。显式 `flush` 是确定性接收纪元边界；同一纪元内不论 TCP 并发顺序如何，批次划分与去向一致。

## 3. 回执格式

| 格式 | 必需字段 |
| --- | --- |
| h1 | `h`, `batch_id`, `destination`, `output_digest` |
| h2 | h1 + `client_bucket`, `weight_bp`, `duration_ms` |
| h3 | h2 + `sample_ids`, `backend_param_digest`, `merge_seq`（可选 `backend_param_version`） |

摘要必须为 `sha256:` + 64 位小写十六进制；h3 样本集合非空且唯一；耗时非负。未知字段进入 `extras`，不会丢弃。缺少必需字段时返回 `E_FORMAT`，并把未知字段列入 `pending_candidates`。相同 `(batch_id, merge_seq)` 在集合中或归并输入中重复均为 `E_FORMAT`。

h1/h2 缺省归并序号分别为 0/1；h3 显式携带。历史 h3 若未携带参数版本，可由归并上下文中的摘要→版本映射补齐。

## 4. 归并规则

候选展示顺序：

```text
(merge_seq ASC, backend_param_version DESC, receipt_identity ASC)
```

采纳规则：

1. 取最高 `merge_seq`。
2. 在该序号内取最高参数版本。
3. 仍相同则取回执身份摘要字典序最小者。

由于重复 `(batch_id, merge_seq)` 被禁止，步骤 2/3 是防御性稳定决胜。被否决候选完整保留，记录：

- `superseded_by_higher_merge_seq`
- `lower_param_version`
- `stable_digest_tie_break`

同时记录去向/输出摘要是否不一致，以及被否决与采纳样本集合的对称差。结论包含去向、参与样本集合、输出摘要、参数摘要与版本、归并序号。输入数组顺序打乱不改变结论摘要。

## 5. 定点口径

会话参数 `scale_exp=k` 定义 `S=10^k`，只允许 `0..9`。

批内值：

```text
duration_q = raw_duration_ms * S
samples_q = raw_samples * S
```

整数毫秒与样本数均精确，量化误差为 0。

规则聚合不先量化均值，而是先累加原始整数：

```text
avg_duration_q = trunc_toward_zero(sum(raw_duration_ms) * S / batch_count)
avg_samples_q = trunc_toward_zero(sum(raw_samples) * S / batch_count)
```

误差口径：

- 单值/规范聚合：真实值位于 `[q/S, (q+1)/S)`，误差 `<1/S`。
- 若把 N 个独立截断的批均值再聚合：最坏误差上界 `<N/S`。
- 账本同时保存原始整数和、批次数、`duration_q`、`samples_q`，可双向复核。

候选比较：两个规范均值各有 `<1` 个定点单位误差，合成保守界为 2。差值 `<=2` 时标记 `indistinguishable=true`、`error_affected=true`，按批次标识字典序决胜；否则选择更低的定点平均耗时。

## 6. SQLite 账本

表：`meta`、`batches`、`receipts`、`cursor`。

每个批次在同一 `BEGIN IMMEDIATE` 事务内：

1. upsert 批次；
2. 清除该批旧回执并写入当前回执集合；
3. 存储归并输入、结论、路由输入、定点统计与预览行；
4. 更新进度游标；
5. 计算批次摘要与整本账内容摘要，更新 `meta`。

`meta.ledger_format_version` 为整数。`content_digest` 覆盖格式版本与按 `batch_index` 排序的批次摘要。读取时逐批重算摘要；任意篡改报 `E_LEDGER`。

## 7. 快照与恢复

快照 manifest 包含：

- 后端集合版本（声明 + 参数摘要/版本）
- 规则版本
- 样本版本
- `scale_exp`
- 后端参数摘要列表

快照 payload 包含：

- 每条规则的稳定/灰度批数、样本数、原始耗时和、定点均值与误差界
- 逐批快速预览
- 归并结论
- 进度游标
- 源账本格式版本与内容摘要

快照顶层 `content_digest` 覆盖格式版本、manifest 与 payload。打开时：

- 摘要不符：保存 `.rejected.json`，整体写回新快照，抛 `E_SNAPSHOT`。
- manifest 任一版本/缩放字段不符：同样整体替换，并稳定列出 `field/snapshot/current`。
- 一致：直接返回 payload 作为快速视图。

恢复重放从快照 `progress.completed_batches` 后开始，逐批重新执行分流、归并、原始整数→定点转换，并对账本字段逐项比较。首个差异报 `E_REPLAY`（批次 + 字段）。

## 8. 快速视图与权威重算

快速视图只读快照，不调用账本重放。权威重算逐批：

1. 由账本路由输入重算桶与去向；
2. 由账本批次标识与样本重算批次身份；
3. 解码并归并保存的回执；
4. 从原始整数重算定点值；
5. 比较存储结论、预览行和规则统计。

快照任一预览项或统计项与权威结果不一致，首个批次报 `E_PREVIEW`。因此快速视图永远不会被伪装成权威事实，状态栏明确显示当前来源。

## 9. 错误码

| 错误码 | 场景 |
| --- | --- |
| `E_SPEC` | JSON 声明、样本、端口、维度、权重、缩放参数非法；首个字段定位 |
| `E_FORMAT` | 回执版本、必需字段、摘要、耗时、重复序号等格式错误 |
| `E_LEDGER` | 账本格式版本、内容摘要、事务或存储错误 |
| `E_SNAPSHOT` | 快照格式、内容摘要或版本/manifest 不一致；已整体替换 |
| `E_PREVIEW` | 快照快速视图与账本权威重算不一致 |
| `E_REPLAY` | 恢复后逐批重放与账本记录不一致 |
| `E_IO` / `E_RUNTIME` | 文件、端口、子进程或运行期故障 |

