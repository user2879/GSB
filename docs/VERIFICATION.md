# 实际验证记录

验证日期：2026-11-20  
工作区：`/home/zsl/feedback/uidfc5909d3-default-302`  
执行环境：WSL2，Linux x86_64，Python `3.11.0rc1`，pip `26.3`，NumPy `2.4.6`，SQLite `3.37.2`

说明：本执行镜像的系统 CPython 未包含 `tkinter`，且当前用户无免密系统包安装权限；CLI、SQLite、子进程、测试与全部核心功能不依赖 Tk。Windows 官方 CPython 安装时勾选 tcl/tk 即可使用可选 GUI。本实现的定点核心不导入 NumPy，但已按准备项安装。除安装上述 Python 包外，运行验证未访问公网目标。

## 1. 规范与单元/集成测试

命令：

```bash
python3 -m unittest discover -s tests -v
```

结果：

```text
Ran 22 tests in 16.4s
OK
```

覆盖项：

```text
FixedPointTests
  floor quantization / single-value bound / comparison bound
  input-order-independent stable rank
SpecValidationTests
  valid sample
  repository/packaged sample identity
  E_CYCLE with closed cycle path
  E_SPEC invalid TTL path/field
  E_SPEC duplicate cache-key field
  E_SPEC unknown dependency
  E_CYCLE introduced by source update history
  E_GRID invalid exponent
StatsTests
  two-level derived weights and conservative bounds
  time-indexed source topology controls batch-derived weights
  cross-node 333/1000 + 1000/1000 + 500/1000 -> aggregate 499/1000, bound 2
  node-order-independent attribution
  pairwise difference within bound annotation
IntegrationTests
  full 3-batch session, verify, snapshot replacement, replay
  E_SNAPSHOT expected-digest mismatch
  snapshot read-only trigger
  E_REPLAY first mismatch at batch 2
  content-change replay reason
  kill -9 style node restart from persistent ledger only
  E_STALE stale hit
  E_PORT occupied source port
DemoAndCliTests
  three demo sessions
  E_CYCLE / E_GRID / E_SNAPSHOT injections
  CLI error exit code
```

## 2. 三会话演示

命令：

```bash
python3 -m cdn_edge demo \
  --ledger /tmp/cdn-final2-demo.i2bHUW/demo.sqlite3 \
  --port-base 28800
```

退出码：`0`

### 2.1 冷启动填充

```text
phase=cold-fill grid=10^3 batches=1 requests=6
batch 1: hit_rate=0, version_consistency=1000, remaining_ttl_units=2833
source_counts: cache_hit=0, expired_revalidated=0, miss_origin=6, stale_hit=0
```

### 2.2 内容版本更新触发派生失效

`base-v1 -> base-v2` 后，派生 `/bundle.js` 与 `/all.js` 的有效 ETag 也发生变化。

```text
phase=version-update grid=10^3 batches=3 requests=18
batch 1: hit=0     vc=1000  ttl=2833  miss_origin=6
batch 2: hit=1000  vc=499   ttl=1233  cache_hit=5, expired_revalidated=1
batch 3: hit=666   vc=1000  ttl=3166  expired_revalidated=4, miss_origin=2
```

批次 2 的版本一致率：

```text
三个节点定点比率:
  edge-a = 333/1000
  edge-b = 1000/1000
  edge-c = 500/1000
按请求数 3/1/2 聚合:
  (333*3 + 1000*1 + 500*2) // 6 = 499
精确值:
  3 consistent / 6 requests = 500/1000
误差:
  1 fixed unit
声明上界:
  2 fixed units
```

批次 2 有 3 个 TTL 内旧版本命中：`r007`、`r008`、`r011`。  
批次 2 的 `r012` 与批次 3 的 `r013/r014/r016/r017` 走 `expired_revalidated`，条目版本从 1 增至 2。

### 2.3 缩放因子扰动

```text
phase=grid-perturb grid=10^2 batches=3 requests=18
batch 1: hit=0   vc=100  ttl=283
batch 2: hit=100 vc=49   ttl=123
batch 3: hit=66  vc=100  ttl=316
```

`10^3` 下的 `499/1000` 在 `10^2` 下表现为 `49/100`，符合统一网格下取整。

### 2.4 多级派生权重

`version-update` 会话的 `/all.js`：

```json
{
  "normalized_fixed_sum_units": 998,
  "normalization_shortfall_units": 2,
  "max_error_bound_units": 3,
  "leaf_weights": {
    "/catalog/base.json": 266,
    "/catalog/theme.json": 399,
    "/promo.txt": 333
  }
}
```

精确叶权重：

```text
base  = 2/3 * 2/5 = 4/15
theme = 2/3 * 3/5 = 2/5
promo = 1/3
```

逐级整数下取整后总和为 `998/1000`；短差 2 个单位被显式记录，保守误差上界为 3 个单位。主导误差段稳定指向 `derive:/all.js->/bundle.js` 对应叶路径。

## 3. 独立账本验证

命令：

```bash
python3 -m cdn_edge verify --ledger /tmp/cdn-final2-demo.i2bHUW/demo.sqlite3
```

结果：

```text
verify_ok=true
batches=7
snapshots=10
active_snapshot_batch=1
active_cache_entries=6

checks:
  batch_metrics_vs_request_records=ok
  error_bounds_and_dominant_sources=ok
  normalized_metric_rows=ok
  snapshot_content_digests=ok
  immutable_snapshot_references=ok
  per_request_consistency=ok
```

## 4. 快照整体替换

演示将活动状态从 `grid-perturb` 的最终批次整体替换为 `version-update` 的第 1 批次快照：

```json
{
  "whole_replacement": true,
  "incremental_merge": false,
  "mismatches": ["content_digest", "grid"],
  "active_cache_entries": 6,
  "old_snapshot_preserved_readonly": true
}
```

直接 SQL 尝试修改快照会触发：

```text
sqlite3.IntegrityError: snapshot_is_read_only
```

## 5. 重放校验

```text
genesis snapshot -> batches [1,2,3], requests 18, all match
batch-1 snapshot -> batches [2,3], requests 12, all match
```

人为篡改批次 2 的 `edge-a.hit_rate.value_units` 后，重放返回 `E_REPLAY`：

```json
{
  "code": "E_REPLAY",
  "path": "session=sample-session/batch=2/node=edge-a",
  "field": "$.nodes.edge-a.metrics.hit_rate.value_units",
  "details": {
    "first_inconsistency": {
      "batch_id": 2,
      "node_id": "edge-a"
    },
    "basis": "recomputed fixed-point statistics and error bounds differ from the immutable snapshot lineage"
  }
}
```

内容版本变化时 `replay --spec changed_spec.json` 返回：

```text
code=E_REPLAY
field=content_digest
reason=content_digest_mismatch
not_reproducible_basis=the request sequence was recorded against a different effective content-version set
```

## 6. 异常注入退出码

```text
$ python3 -m cdn_edge demo --inject cycle
E_CYCLE  path=content[path=/all.js].derivation  field=inputs  exit=3

$ python3 -m cdn_edge demo --inject grid
E_GRID   path=grid                              field=power   exit=5

$ python3 -m cdn_edge demo --inject snapshot
E_SNAPSHOT path=snap-...                        field=content_digest exit=6

$ python3 -m cdn_edge demo --inject replay
E_REPLAY path=snap-...                          field=grid  reason=grid_mismatch exit=7

$ python3 -m cdn_edge demo --inject port
E_PORT   path=session.source_port               field=port  exit=4
```

额外验证：

- 节点进程 `kill()` 后立即在原端口重启，缓存条目与请求计数仅从节点 SQLite 账本恢复；工作目录不存在任何退出报告。
- 源站被强制停止后，过期条目在 `allow_stale=true` 下返回 `stale_hit`、`E_STALE`、剩余 TTL 0。
- 被占用端口在任何子进程启动前返回 `E_PORT`。

## 7. 纯命令行闭环

在独立随机高位端口上执行：

```bash
python3 -m cdn_edge run --spec spec.json --ledger ledger.sqlite3
python3 -m cdn_edge verify --ledger ledger.sqlite3
python3 -m cdn_edge snapshots --ledger ledger.sqlite3 --session sample-session
python3 -m cdn_edge replay --ledger ledger.sqlite3 --snapshot-id snap-...
python3 -m cdn_edge restore --ledger ledger.sqlite3 --snapshot-id snap-... --mode replace
```

结果：

```text
run:       rc=0, batches=3
verify:    rc=0, all six checks=ok
snapshots: rc=0
replay:    rc=0, batches [1,2,3], requests 18
restore:   rc=0, whole_replacement=true, cache_entries=6
```
