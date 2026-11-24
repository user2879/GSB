# 设计说明

## 1. 模块边界

| 模块 | 职责 |
| --- | --- |
| `config.py` | 主配置、请求流、归并规则加载与业务校验 |
| `receipts.py` | r1/r2/r3 文件读取、版本/字段布局/类型校验、失败隔离 |
| `models.py` | 统一回执、工作进程、请求、批次、裁决、资源、调度数据模型 |
| `conversion.py` | 双向转换、显式缺失标记、降级旁路与转换账本 |
| `normalization.py` | 批次内聚合、稳定名次冲突、跨批重复投递裁决 |
| `resources.py` | 整数定点乘加量/利用率、质量摘要、依赖失效判定 |
| `scheduler.py` | 预算和单工作进程累计配额调度 |
| `snapshot.py` | 阶段目录、不可变代、原子当前指针、恢复判定 |
| `worker.py` | NumPy 定点/float64 回环推理子进程 |
| `supervisor.py` | 子进程启动、健康检查、HTTP 批量推理 |
| `pipeline.py` | 阶段编排、线程池、缓存复用、账本/报告输出 |

## 2. 统一回执

`Receipt` 的九个业务字段始终存在。r1 的六个扩展字段、r2 的三个扩展字段均为 `MISSING` 单例，序列化为 `{"$missing": true}`。该表示不把缺失值与合法的整数零、空引用混淆。

解析器接受：

- 单条回执对象；
- 带版本标记的回执列表；
- `{"version":"r3","receipts":[...]}` 信封；
- 版本别名 `r1`/`1`/`receipt-r1`，内部统一为 `r1/r2/r3`。

同一文件所有记录必须使用相同版本；未知字段、缺少字段、类型错误分别产生 `E_RECEIPT_LAYOUT` / `E_RECEIPT_FIELD_TYPE`。单条失败不阻止同文件或后续文件中的有效记录。

## 3. 转换与旁路

转换以 `sidecar_key = batch_id|request_id|output_digest` 标识逻辑回执。它只由 r1 仍可读取的字段构成，因此 r3→r1 落盘再读取后仍能定位旁路。

r3→r1 时：

1. 所有被目标版本丢弃的字段存入 sidecar；
2. 目标内部表示将丢弃字段置为 `MISSING`；
3. 转换账本记录 `downgrade`、`downgraded_fields`、sidecar key 与完整性；
4. 序列化只写 r1 字段。

r1→r3 时：

1. 优先从 sidecar 恢复字段；
2. 无可恢复值时写显式缺失标记；
3. 转换账本记录 `restored_fields` 与 `irreversible_fields`；
4. `completeness` 变为 `complete` 或保持 `partial`。

格式往返不会改变 `batch_id/request_id/output_digest`，恢复字段也逐一相等，所以归属裁决的候选键、参数版本、耗时和稳定名次不变。测试 `test_format_roundtrip_does_not_change_ownership_decision` 对裁决对象做逐字段比较。

## 4. 归属裁决

### 批内

先按 `(batch_id, effective_worker_id)` 聚合。r1/r2 没有 worker 字段时，使用请求流中该批次的配置工作进程作为上下文，但不改写回执内部缺失值。

同一批同一请求出现多个工作进程时：

- 若双方稳定名次不同，记录 `RankConflict(worker_a, worker_b, rank_a, rank_b)`；
- 按工作进程标识字典序升序选择胜者；
- 冲突双方均持久化在账本中。

### 跨批

同一请求出现在多个批次时：

1. 若存在 r3 `overwritten_ref`，建立“被覆盖 → 覆盖”边；被覆盖候选淘汰；
2. 在终端候选中按 `duration_ms` 升序，缺失耗时排后，再按批次/工作进程/UID 稳定排序；
3. 无覆盖引用时，按参数版本降序、批次标识升序、工作进程升序、UID 升序；
4. `basis` 记录规则名、全部候选 UID、覆盖边、被覆盖 UID、胜者耗时/版本/工作进程。

覆盖引用可写 `receipts/file.json#0` 形式的 UID，也可写 `batch:request:worker:rank`。循环覆盖回退到无覆盖规则并记录 `overwrite_cycle_fallback`。

## 5. 资源与失效

资源计算全部使用 Python 整数：

```text
base_parameters  = input_dim*hidden_dim + 2*hidden_dim
macs_per_request = 2*base_parameters
macs             = request_count*macs_per_request
quantized_macs   = ceil(macs*quantization_bits/8)
utilization_ppm  = min(1_000_000,
                       request_count*1_000_000/(batch_size*max(1,batch_duration_ms)))
```

批次耗时为权威请求中已观测耗时的最大值。没有观测值时，耗时和利用率都是缺失标记，避免把 r1 缺失耗时当成 0 ms。

缓存项包含：

```text
worker_id, worker_parameter_version, input_hash,
normalization(receipts+candidates+conflicts), decision_hash, resources
```

失效判定：

- 工作进程参数版本变化：只失效该工作进程的批次及其请求；
- 回执或请求语义变化：只失效对应批次；
- 模型维度/量化口径/批大小变化：全部资源重算，批内归一结果仍可复用；
- 归并规则变化：重新执行全局裁决；批内归一可复用，资源仅在 `decision_hash` 变化时重算。

全量重建与增量的最终 `ledger.json` 都只表达最终输入的函数，因此逐字节一致。真实缓存命中/重算记录在诊断报告的 `actual_invalidation` 中，并用于证明未受影响条目没有重新计算。

## 6. 调度器

调度单位是“请求归一处理点”。批次按以下顺序处理：

1. 上一快照的待办批次优先，按 `batch_id` 排序；
2. 其余批次按 `batch_id` 排序。

每批先检查工作进程累计配额。若 `已用 + 本批点数 > 配额`，不扣预算，批次进入 `quota_blocked` 并产生 `E_WORKER_QUOTA_EXCEEDED`。预算不足整批时，不进行部分提交，批次进入待办并产生 `E_NORMALIZE_BUDGET_PENDING`。

每个被接受的批次逐点写入 trace：`point_index`、批次剩余点数、剩余预算、工作进程已用点数和配额。线程池只负责已获票批次的计算，不参与预算/顺序裁决。

## 7. 快照提交

```text
snapshots/
  CURRENT.json                      # 原子替换的指针
  generations/g00000001/
    parse/payload.json
    ownership/payload.json
    resources/payload.json
    ledger/payload.json
    cache.json
    manifest.json
```

阶段先在 `.staging-*` 中建立 `.stage.tmp` 目录，写完后 `os.replace` 到同名阶段目录。全部阶段和 `cache.json` 完成后，整个 staging 目录改名为新代，最后原子替换 `CURRENT.json`。旧代不逐文件覆盖，可直接审计。

恢复时比较当前清单中的配置哈希和回执指纹；同时记录代理版本和待办状态。即使配置变化，新运行也从上一代缓存按依赖边界增量恢复，而不是复制/覆盖旧代文件。

## 8. 并发确定性

线程池任务键：

```text
normalize|batch_id|worker_id
resources|batch_id|worker_id
```

任务提交前按键排序，完成后按键取值并再次按业务键排序；所有共享结构只在主线程合并。失败列表最终按来源文件和序号排序。因此并发度不改变账本、裁决、指标、待办或失败顺序。

## 9. 故障演练

正常回执与独立 `fault_receipts/` 一同解析；注入配置把相对文件名映射到注入编号。失败记录携带 `injection_id`，报告汇总 `hit=true/false`。四类注入均只影响对应记录/文件，其余有效回执继续进入账本。故障演练不提交主快照，报告写入 `reports/fault_drill/`。
