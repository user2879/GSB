# 技术说明：定点误差、派生传播、账本与恢复

## 1. 运行拓扑

```text
process: source_worker
  127.0.0.1:<source_port>
        ^ JSON line: get / versions / update
        |
  +-----+------+-------------+
  |            |             |
edge-a       edge-b        edge-c
node ledger  node ledger   node ledger
        \      |      /
       orchestrator / stats
              |
      SQLite diagnostic ledger
      batches + immutable snapshots
```

每个节点是独立进程，拥有独立 SQLite 状态账本。编排器通过回环 TCP 发送请求并读取节点报告。所有内容均来自 `samples/sample_spec.json` 与进程内存/本地账本，不发起公网连接。

## 2. JSON 行协议

一行一个 JSON 对象，响应也是一行。请求示例：

```json
{"op":"fetch","seq":7,"request_id":"r007","path":"/catalog/base.json","at_ms":1900,"key":{"path":"/catalog/base.json","query":"q=main"},"allow_stale":true}
```

节点响应关键字段：

```json
{
  "ok": true,
  "node_id": "edge-a",
  "hit_source": "cache_hit",
  "etag": "base-v1",
  "remaining_ttl_ms": 100,
  "entry_version": 1,
  "written_at_ms": 0,
  "written_mono_ms": 123456789,
  "monotonic_base_ms": 123456789,
  "stale": false,
  "error_code": null
}
```

`monotonic_base_ms` 是节点持久化的单调时钟毫秒基准。确定性的虚拟时间 `at_ms` 叠加到该基准上，得到条目写入的单调时刻。节点重启时基准和条目时间均从 SQLite 恢复，因此 TTL 计算不回退。

源站操作：

- `get`: 按 `path` 和 `at_ms` 返回有效内容版本；
- `versions`: 返回某时刻所有路径的 ETag 集合与内容摘要；
- `update`: 追加带 `effective_from_ms` 的版本历史；
- `ping` / `shutdown`: 进程控制。

派生内容 ETag 为：

```text
ETag(derived) = declared_etag ~ SHA256(weighted_derived_bytes)[0:16]
```

所以依赖内容变化即使没有显式更新派生路径，派生 ETag 也会变化。

## 3. 统一定点网格

会话参数 `k` 定义：

```text
S = 10^k
Q(x) = floor(S * x), x >= 0
```

`k` 必须是 `0..12` 的整数。所有指标报告 `value_units=Q(x)`、`grid.scale=S` 和 `truncation=floor_toward_negative_infinity`。

### 3.1 单值量化误差

```text
0 <= S*x - Q(x) < 1
```

因此固定单位误差界为 1 个 grid unit，即 `< 1/S` 实数误差。

### 3.2 比率与 TTL

```text
hit_rate_units              = floor(hit_count * S / request_count)
version_consistency_units   = floor(consistent_count * S / request_count)
remaining_ttl_units         = floor(sum_remaining_ms * S / (1000 * request_count))
```

分子、分母及乘加均为整数，没有任何浮点中间值。

### 3.3 比较误差界

若两个真实比率分别量化为 `qa`、`qb`：

```text
|qa - qb| <= 2
```

在比较误差界内。超过 2 个 fixed unit 才能确定真实值排序不同。工具对每一对同类型节点指标输出 `within_error_bound`、`bound_units=2` 和稳定决胜顺序。

误差界内的排序规则：

1. 按固定值降序形成候选组；
2. 组内最大值与最小值之差 `<= 2` 时视为一个误差主导组；
3. 组内按节点 ID 或路径 ID 字典序决胜，绝不使用输入/线程完成顺序。

## 4. 跨节点聚合误差传播

节点指标先独立量化：

```text
q_i = floor(hit_i * S / n_i)
```

聚合值使用整数加权：

```text
q_agg = floor(sum_i(q_i * n_i) / N), N = sum_i(n_i)
```

令真实比率为 `x_i`，节点量化误差为 `e_i in [0,1)` 个 fixed unit：

```text
q_agg - S*x_agg in [0, 2)
```

因此聚合误差界为 2 个 fixed unit：

- 节点量化误差加权和 `< 1` unit；
- 最终带权整数除法下取整 `< 1` unit。

输出 `segments` 包含每个节点权重 `n_i/N` 对误差预算的贡献，以及最终的 `aggregate:final_truncation`。求和顺序不影响结果，因为节点先按 ID 排序、整数加法可交换。

## 5. 多级派生权重

每个派生内容声明整数系数：

```text
P = sum_i(c_i * D_i), c_i > 0
w_i = c_i / sum_j(c_j)
```

叶路径的精确权重用 `fractions.Fraction` 推导。定点传播逐级使用：

```text
c_unit_i = floor(c_i * S / sum_j(c_j))
q_weight(P, leaf) = sum_i floor(c_unit_i * q_weight(D_i, leaf) / S)
```

权重误差保守上界递推：

```text
B(P, leaf) = sum_i[
    w_i * B(D_i, leaf)
  + (w_i_leaf + B(D_i, leaf)) / S
  + 1 / S
]
```

三项分别覆盖：

1. 子链已有误差沿系数放大；
2. 子权重截断误差；
3. 当前层整数乘除下取整。

最终以 `ceil(B*S)` 报告固定整数上界。每个依赖段输出：

- `contribution_units`：该段对固定单位误差界的贡献；
- `share_units`：归一化占比，`S` 表示 100%；
- `dominant_error_segment`：最大贡献段，平局按来源字典序。

样本 `/all.js` 的两级链：

```text
/all.js = 2/3 * /bundle.js + 1/3 * /promo.txt
/bundle.js = 2/5 * base + 3/5 * theme
```

在 `S=1000`、`k=3` 时，量化后叶权重为：

```text
base  = 266/1000
theme = 399/1000
promo = 333/1000
```

短差 2 个单位由逐级下取整造成，报告 `normalization_shortfall_units=2`，并给出 3 个固定单位的上界。

## 6. 请求版本一致性

每个请求记录：

- `hit_source`: `cache_hit`、`expired_revalidated`、`miss_origin`、`stale_hit`；
- 返回 ETag；
- 同一 `at_ms` 下源站权威 ETag；
- `version_consistent = returned_etag == authoritative_etag`；
- 剩余 TTL、条目版本号、`stale` 与 `E_STALE` 标记。

内容更新后，TTL 内缓存仍可能返回旧版本；这不是协议错误，但会降低版本一致率。发生真实过期时：

1. 节点向源站重校验；
2. 不论 ETag 是否相同，整体替换条目内容/版本/TTL；
3. `entry_version += 1`；
4. 返回 `expired_revalidated`。

若源站不可达且 `allow_stale=true`，过期条目可作为 `stale_hit` 返回，但必须带 `E_STALE`、零剩余 TTL，并计入版本一致率。

## 7. SQLite 事务与不可变快照

核心表：

- `sessions`: 原始 spec、更新列表、初始内容摘要、网格；
- `batches`: 批次统计、误差摘要、内容摘要、游标；
- `batch_requests`: 逐请求命中记录；
- `node_metrics`: 每节点/聚合指标、值、误差界、主导来源、分段 JSON；
- `snapshots`: 不可变内容版本集、节点缓存状态、结论、规格和游标；
- `active_state` / `active_cache`: 最近恢复或完成批次的活动状态。

每个批次在同一个 `BEGIN IMMEDIATE ... COMMIT` 中写入：

```text
batch row
+ all batch_requests
+ all node_metrics
+ snapshot row
+ active_state/active_cache whole-state update
```

`snapshots_no_update` 与 `snapshots_no_delete` 触发器拒绝修改或删除任何快照，返回 `snapshot_is_read_only`。

快照身份为：

```text
SHA256(session_id, batch_id, kind, cursor, content_digest, grid)[0:20]
```

打开快照时同时校验：

- 内容版本集重新计算的摘要；
- 快照 ID 与版本/网格/游标坐标；
- payload 内摘要。

任何摘要不符返回 `E_SNAPSHOT`。

## 8. 整体替换恢复

`restore --mode replace` 永远执行整体替换，不做增量合并：

1. 删除活动缓存与活动结论；
2. 用快照内节点状态重建 `active_cache`；
3. 用快照批次结论替换 `active_state.conclusions_json`；
4. 旧快照与历史批次不修改。

若当前内容摘要或网格与快照不同，报告 `mismatches` 与 `whole_replacement=true`。`--mode strict` 遇到同样差异直接返回 `E_SNAPSHOT`。显式传入错误 `--expected-content-digest` 也返回 `E_SNAPSHOT`。

## 9. 重放校验

`replay`：

1. 读取并校验快照；
2. 从快照 payload 恢复 spec、源更新历史、网格和节点缓存状态；
3. 从 `cursor.next_request_index` 开始重新启动本地源站/节点并驱动请求；
4. 按规定网格重新计算批次统计、误差界、主导误差段和派生权重；
5. 与账本逐批、逐节点、逐字段比较。

任何差异返回 `E_REPLAY`，`details.first_inconsistency` 明确包含：

```json
{
  "batch_id": 2,
  "node_id": "edge-a",
  "field": "$.nodes.edge-a.metrics.hit_rate.value_units",
  "expected": 667,
  "actual": 666
}
```

若网格或内容版本集不可复现，错误详情给出 `reason`、两侧值和 `not_reproducible_basis`，而不是笼统失败。

## 10. 确定性边界

- 请求顺序由 JSON 数组和 `seq` 固定；
- 内容版本由虚拟时间轴选择，不依赖本地墙钟或线程完成顺序；
- 节点每个请求先提交 SQLite 再响应；
- 节点、路径、叶权重和分段输出全部按 ID 排序；
- 聚合只使用整数乘加；派生精确值使用有理数，不经过浮点；
- 强制终止节点不产生“退出报告”，重启只加载已有节点账本。

因此同一 spec、网格、内容版本集和请求序列可在无端口冲突的机器上稳定重放。
