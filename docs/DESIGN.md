# 设计与确定性口径

## 1. 分层结构

| 模块 | 职责 |
|---|---|
| `spec.py` | JSON 强校验；用轻量 tokenizer 为字段路径保留源行号 |
| `records.py` | v1/v2/v3 检测、编解码、归一、补齐溯源与规范结论摘要 |
| `scoring.py` / `worker.py` | NumPy 整数定点打分；真实回环 TCP 客户端与子进程服务 |
| `scheduler.py` | Filter、Score、稳定决胜、回执目录、全量与增量子集调度 |
| `invalidation.py` | 参数、节点、Pod、插件、回执引用图的失效传播 |
| `migration.py` | v1/v2 → v3 批事务、检查点、只读备份、逐字段与逐字节校验 |
| `ledger.py` | SQLite 会话快照、内容摘要、整体替换、归档差异、逐 Pod 提交与恢复 |
| `verifier.py` | 只读账本、从零独立 Filter/Score 重算并逐字段核对 |

## 2. 声明校验

输入是一个本地 JSON 对象。字段顺序校验固定，任一错误抛出首个 `E_SPEC`，并带 `field_path`、`source_line`、`source_path`。

强约束包括：

- 节点名和 Pod 标识各自唯一且非空。
- 节点 `allocatable.cpu/memory` 与 Pod `requests.cpu/memory` 必须为正整数（布尔值不算整数）。
- 污点 `effect ∈ {0:NoSchedule, 1:PreferNoSchedule, 2:NoExecute}`。
- 四个插件均需声明 `weight_myriad`，非负且总和恰为 `10000`。
- `scale_exponent` 是 `0..3` 的整数，实际因子为 `10**e ∈ {1,10,100,1000}`。
- 服务只允许 `127.0.0.1`；节点必须且只能出现在一个打分端口组。
- 标签选择器、Pod 反亲和拓扑键、容忍操作符均受白名单约束。

## 3. 记录格式与归一

### v1

```text
v1 + pod_id(18) + node_name(18) + total_score(6,right) + veto_code(4,right)
```

总宽 48。短于 48 视为定宽字段不足，报 `E_FORMAT`；长于 48 视为宽度声明不一致，报 `E_WIDTH`。

### v2

必需字段：`format_version, pod_id, node_name, plugin_scores, total_score, veto_code, tie_break`。插件分项和总分均为浮点；`tie_break` 可为逗号字符串或数组。

### v3

在 v2 基础上要求：`parameter_version`、`scale_exponent` 或 `scale_factor`、逐插件 `plugin_contributions`、`error_bound`、整数 `total_score`。可携带 `plugin_weights_myriad`；缺失时从当前声明补齐并标注来源。

### 补齐规则

| 来源 | 字段 | 规则 |
|---|---|---|
| v1 | 插件分项/贡献 | 显式零值；原文无该信息，不反推 |
| v1 | 参数/缩放/权重 | 使用归一调用方声明的参数 |
| v1 | `tie_break` | `total_score,node_name` |
| v2 | 插件分项 | half-up 量化为整数 |
| v2 | 贡献/误差界 | 由当前权重和缩放因子按定点公式计算，误差界取保守值 |

每条归一记录都携带 `filled_fields` 和 `basis[field] = {rule, source}`。规范结论字节由排序键固定的紧凑 JSON 生成，不含补齐说明之外的运行期字段。

## 4. Filter 与 Score

候选节点按名称排序，依次执行：

1. `taint_toleration`：`NoSchedule/NoExecute` 未容忍立即拒绝；`PreferNoSchedule` 只扣分。
2. `node_affinity`：所有硬标签项必须满足。
3. `pod_anti_affinity`：与已确定绑定 Pod 的硬选择器和拓扑键冲突即拒绝。
4. `resource_fit`：Pod CPU/内存请求不得超过节点可分配值。

每个节点的 `filter_trace` 保存 `order/plugin/result/reason`；拒绝后不再执行后续步骤。

通过 Filter 的节点向子进程发送插件特征。Worker 使用 NumPy `int64` 数组计算：

```text
raw_i ∈ [0,100]
weighted_numerator_i = raw_i × weight_myriad_i × scale_factor
contribution_i = floor(weighted_numerator_i / 10000)
total_fixed = min(sum(contribution_i), INT64_MAX)
total_score = floor(total_fixed / scale_factor)
error_bound = ceil(plugin_count / scale_factor)
```

所有输入非负，因此 `//` 即向零/向下截断。单项分子先饱和到 `INT64_MAX`，总和再次饱和。返回值同时包含原始分、逐插件贡献、`total_fixed`、`total_score`、参数版本、缩放因子、误差界和单调时钟毫秒耗时。

决胜键：

```text
(-total_fixed, -raw_taint, -raw_node, -raw_anti, -raw_resource, node_name)
```

节点枚举顺序被显式排序消除；毫秒耗时只存在于回执审计材料，不进入稳定结论摘要。

## 5. 子进程协议

主进程为每个端口组启动：

```bash
python -m affinity_scheduler.worker --host 127.0.0.1 --port <declared>
```

Worker 成功 bind/listen 后输出现实端口握手 JSON。主进程用启动 select 超时、TCP connect 超时和逐回执 deadline 三重超时控制。请求和回执均为单行规范 JSON。

错误边界：

- 启动前探测端口被占用或 worker 绑定失败：`E_PORT`。
- 握手/连接/发送/读取超时：`E_TIMEOUT`。
- 回执缺字段、身份或版本不匹配、插件集合不完整：`E_REPLY`。

## 6. 失效传播

状态文件保存当前 spec 指纹、逐个 Pod 的完整结论和回执目录。参数版本必须单调不减。

- 节点集合或内容变化：全部 Pod 失效，因为候选集合、硬约束或资源分项可能变化。
- 缩放因子变化：全部 Pod 失效，定点分母变化。
- 插件权重变化：只标记在任一候选回执中该插件分项非零的 Pod。
- Pod 声明变化：该 Pod 失效；若声明集合变化，所有带 Pod 反亲和的 Pod 进入闭包，因为绑定转移可能改变冲突。
- 参数版本只变化而有效参数未变：输出 metadata 传播步骤，不强制重算。

受影响节点是失效 Pod 全部候选回执节点的并集，再并入变化节点。步骤按 Pod 标识、来源和原始序号稳定排序后重新编号。

增量调度保留未失效 Pod 的完整结果与回执；失效 Pod 重新执行 Filter/Score。`verify-incremental` 同时执行从零全量路径，并比较所有 Pod 的稳定结论字节。

未失效 Pod 的外层 `parameter_version` 保留为“产生该回执的版本”，并额外记录 `reused_receipts=true` 与 `receipt_parameter_versions`。账本全局元数据保存当前有效参数版本；这样既不会伪造旧回执的版本，也不会把未变化的 Pod 列入账本替代清单。

## 7. 迁移状态机

1. 读取并检测混合记录版本。
2. 确认原始文件 SHA-256 和参数声明，生成确定性 batch ID。
3. 复制原文件到只读备份，校验内容后设为 `0444`。
4. 按 `(source_format_version, source_line)` 排序，逐条编码 v3。
5. 重新解码 v3，比较节点、插件分项、贡献、否决码、总分、误差界和决胜依据；首差异报 `E_MIGRATE`。
6. 每 `batch-size` 条在一个 SQLite 事务内写入迁移行并更新检查点。
7. 全部提交后按序生成文本，原子 `os.replace` 原文件，最后把批次置为 `complete`。

测试专用 `--abort-after-batches N` 在提交后直接 `os._exit(137)`。恢复时只使用数据库中的 checkpoint/target rows，原文件仍为旧版本。

## 8. 账本与恢复

核心表：

- `ledger_meta`：活动快照、版本集合、参数版本、缩放因子、游标、内容摘要。
- `ledger_entries`：当前逐 Pod 摘要、状态、规范结论和摘要。
- `ledger_snapshots`：会话快照元数据。
- `ledger_snapshot_entries`：上一活动快照的逐 Pod 内容摘要归档，用于稳定差异。
- `ledger_replacement_events`：新增/删除/被替代事件。

`sync_snapshot` 比较版本集合、声明的内容摘要、当前行实际摘要和归档摘要。不一致时在一个事务内删除并重建全部条目，不做增量合并；差异按 Pod 标识排序稳定输出。

调度开始时建立 pending session，将全部计划 Pod 标为 `unknown`；每完成一个 Pod 单独事务提交并推进 pending cursor。进程被终止后，恢复命令只读取已提交行，保留未知 Pod，要求重跑，并明确不能生成退出报告。

## 9. 独立核对

`verify` 以 SQLite `mode=ro` 打开账本，读取原始 spec/records，重新进行：

- 记录归一与格式版本集合核对。
- 独立四阶段 Filter 轨迹构建。
- 通过同一真实回环 Worker 做定点打分和稳定决胜。
- 与账本中每个 Pod 的结论逐字段比较。

首个差异抛 `E_VERIFY`，携带 Pod 标识、节点名、字段和两侧值。

## 10. 可复现性边界

- 结论摘要只包含业务字段，不含 wall clock、单调时钟耗时、进程 PID 或套接字局部信息。
- 排序在所有节点、Pod、插件、迁移记录、账本差异和失效步骤上显式指定。
- Python 浮点只出现在 v2 历史输入；归一后立即量化。v3 及新调度均走整数路径。
- 舍入与截断规则固定在代码和文档中；环境差异不会改变整数结果。
