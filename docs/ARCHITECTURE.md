# 边缘推理分片调度诊断实现说明

本目录是一个完全本地、确定性的边缘推理分片调度诊断实现。节点是真实子进程，节点与调度器之间通过 `127.0.0.1` 上的 TCP JSON-lines 通信；权威状态写入 SQLite。所有演示样本由固定整数种子 `407` 生成，不访问外部账号、站点或数据集。

## 1. 复核入口

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install numpy==1.26.4

python scripts/generate_demo.py --check
python -m edge_shard_diag validate-spec --spec config/demo
python -m edge_shard_diag receipts
python -m edge_shard_diag run
python -m unittest discover -s tests -v
python -m edge_shard_diag acceptance
python -m edge_shard_diag verify-incremental
python -m edge_shard_diag verify-recovery
python -m edge_shard_diag ui --check
```

一键脚本：`scripts/acceptance.sh`。

当前 WSL 镜像缺少 `ensurepip`、`python3-tk` 与 `sqlite3` CLI，且 `sudo` 需要密码。实际环境使用 `python3 -m venv --without-pip .venv` 后通过 `get-pip.py` 引导 pip；SQLite 使用 Python 标准库 `sqlite3`。有 sudo 密码时可执行：

```bash
sudo apt-get update
sudo apt-get install -y python3.11-venv python3-tk sqlite3
```

## 2. 模块地图

- `edge_shard_diag/spec.py`：带行号 JSON 解析、强校验、版本链、DAG、稳定拓扑序和内容摘要。
- `edge_shard_diag/jsonloc.py`：保留对象/字段/数组元素源码行号的 JSON 解析器。
- `edge_shard_diag/aggregate.py`：固定种子样本与 NumPy 分组聚合、摘要和样本快照。
- `edge_shard_diag/receipts.py`：d1 定宽行、d2/d3 JSONL、默认补齐、跨版本采纳排序。
- `edge_shard_diag/node_worker.py`：TCP JSON-lines 节点子进程。
- `edge_shard_diag/tcpio.py`：真实套接字握手、读写超时与 `E_PORT`/`E_TIMEOUT` 映射。
- `edge_shard_diag/nodepool.py`：节点子进程启动、端口就绪探测、日志和回收。
- `edge_shard_diag/scheduler.py`：就绪冻结集、单次调度、线程池、租约续约和排队计时。
- `edge_shard_diag/fixedpoint.py`：整数定点量化、截断/饱和规则、水位误差界和三段归因。
- `edge_shard_diag/diagnostics.py`：可与全量重建逐字段比较的确定性水位/误差投影。
- `edge_shard_diag/invalidation.py`：样本/节点/分片依赖/回执引用传播、字节哈希和字段比对。
- `edge_shard_diag/ledger.py`：SQLite 原子事务、租约/资源槽位、快照、陈旧版本和崩溃恢复。
- `edge_shard_diag/acceptance.py`：三格式、租约、定点、增量、换代、SIGKILL 恢复的端到端验收。
- `edge_shard_diag/ui.py`：Tkinter 演示界面，显示未知分片标记和快照版本列表。

## 3. 错误码

| 码 | 触发点 |
| --- | --- |
| `E_SPEC` | JSON 语法、字段类型/取值、唯一约束、端口字面量、版本链、依赖图、缩放集合不合法 |
| `E_FORMAT` | 未知回执格式版本、d1 宽度不符、JSONL 语法错误 |
| `E_FIELD` | d2/d3 必需字段缺失、类型错误、d3 稳定名次越界 |
| `E_DUP` | d3 重复投递引用不在已知投递/回执引用集合中 |
| `E_LEASE` | 租约乐观版本不符、租约非 ACTIVE、提交结局时租约失效 |
| `E_PORT` | 端口占用、连接拒绝、节点绑定失败、连接关闭 |
| `E_TIMEOUT` | TCP 连接/握手/读取超过显式套接字超时 |
| `E_BUDGET` | 全局资源单位、节点租约名额或节点并发配额不足 |
| `E_DIVERGE` | 增量视图与空白全量重建在某分片某字段不一致 |
| `E_REPLAY` | 强制终止恢复后的账本/游标/重放结果与全量路径不一致 |

所有结构化错误均携带 `code`、`field`、`line`、`source` 和可选 `details`。

## 4. 声明与版本语义

演示声明位于 `config/demo/`：

- `manifest.json`：声明格式版本、节点集合版本、当前容量版本和定点修订版本。
- `nodes.json`：节点 ID、序号、字面量回环地址、唯一端口、并发配额、租约上限、节点自身定点缩放。
- `shards.json`：分片、节点归属、批次、序号区间、资源单位和上游依赖。
- `samples.json`：固定整数种子 `407` 生成的整数样本。
- `capacity_versions.json`：容量版本链 v1 → v2，每个版本完整声明预算和所有节点限制。
- `fixed_revisions.json`：定点修订链 r1=1、r2=10、r3=100。
- `event_stream.json`：严格递增事件流样例。
- `delivery_refs.json`：已知重复投递引用。

`SpecBundle.content_digest` 对声明目录中的文件相对路径和字节计算 SHA-256。任何声明编辑都会改变内容摘要。

## 5. 回执归一

d1 默认宽度为 `4 + 5 + 5 + 16 = 30` 字符，最后一个字段必须等于声明的 `prefix_chars=16`。低版本缺字段按以下规则补齐：

- `capacity_version=0`：未知基线，采纳排序低于所有正容量版本。
- `scale_factor=1`：受支持的最小 `10^0`。
- `stable_rank=-1`：低版本不参与 `1..batch_size` 名次区间。
- `duplicate_refs=[]`：没有可判定的重复覆盖引用。
- `duration_ms=0`、`sample_snapshot=""`：显式零值/空快照。

每条归一结果包含 `format_version`、`source_file`、`source_line`、`supplemented_fields` 和 `supplement_basis`。采纳键为：

```text
(-capacity_version, -format_order[d1=1,d2=2,d3=3], stable_rank, source_file, source_line)
```

即先采纳容量版本更新的结论；容量版本相同时，d3 优先于 d2，d2 优先于 d1。

## 6. 调度、租约与资源不变量

稳定顺序使用 Kahn 拓扑：每一步在所有依赖已满足的分片中选择 `shard_id` 最小者。候选放行和误差临界时的最终决胜均使用该顺序。

调度器每轮先得到不可变的 `freeze_ready_set`。只有上游全部 `COMPLETED` 且 SQLite 中存在非空摘要的分片才能进入集合。每个分片只提交一次 Future；`FROZEN` 分片不会进入候选集。

SQLite 通过三类所有权记录防止重复占用：

- `allocations` 中 `status='HELD'` 的分片有部分唯一索引。
- `resource_owners(capacity_version, unit_index)` 逐个资源单位记录属主，释放/抢占均带 `AND status='FREE'` 或 `AND status='ACTIVE'` 条件。
- `lease_slots(capacity_version,node_id,slot_index)` 逐个租约名额记录属主。

租约包含 `granted_ms`、`expires_ms`、`ttl_ms`、`renewed_ms`、`renew_count`、`lease_version`。到期判断只调用 `time.monotonic_ns()`；测试用真实短 TTL + `time.sleep` 验证，不注入假时钟。续约先向节点子进程发送当前版本做乐观校验，节点匹配后版本加一，随后账本在事务内延长到期时间。版本不符抛 `E_LEASE`，节点和账本都不延长到期。

到期回收与换代撤销竞争同一条 ACTIVE 租约时，只有条件更新 `WHERE status='ACTIVE'` 成功的事务能释放资源单位和租约槽位；另一个路径影响行数为零，因此同一份名额不会被双重裁决。

## 7. 定点水位与误差归因

会话缩放因子 `S=10^k` 必须属于受支持集合。所有非负比例使用：

```text
q(n,d,S) = floor(n * S / d)
remainder = n*S - q*d
```

- 截断方向：非负数向下取整，即 `floor`。
- 比较误差：单个比例小于 1 个定点单位；两个比例之差小于 2 个定点单位。
- 全局保守上界：若聚合 N 个分片截断比例，误差上界为 N 个定点单位；直接聚合预算比例的紧上界为 1。
- 溢出规则：绝对值超过 `2^63-1` 时饱和到 `2^63-1` 或 `-2^63`，并在记录中设置 `saturated=true`。

合成判定包络为：

```text
epsilon_q = 2(量化比较)
          + capacity_generation_q
          + dependency_readiness_q
          + lease_renewal_q(=1+renew_count)
```

判定阈值是：

```text
abs(slack_q) <= epsilon_q 或 abs(level_delta_q) <= epsilon_q
```

命中任一则该放行被标记为 `error_affected=true`，按稳定拓扑序和分片 ID 序选择第一个候选。演示中的 `3/6` 与 `4/6` 在 `S=1` 都量化为 `0`；在 `S=100` 分别为 `50` 和 `66`，差值 `16` 超过包络 `4`，从误差临界变为可判定。

## 8. 失效传播与逐字段一致性

传播种子包括：

- 样本 ID：找到直接引用该样本的分片。
- 节点 ID：找到绑定该节点的分片。
- 容量/缩放换代：所有分片结论进入失效/陈旧集合。
- 回执 ID：直接失效并沿 `duplicate_refs` 中指向已知回执的引用传播。

分片沿反向依赖边传播到全部直接和间接下游。输出中的 `steps` 按确定性顺序记录每一条 `source -> target` 和依据；`invalid_shards` 按稳定拓扑序排列。

增量路径只重算失效集合，且只覆写这些分片的规范 JSON 记录；未失效文件不打开写入，因此 SHA-256 逐字节不变。全量路径在空白目录、空白 SQLite 和新端口上重跑真实 TCP 调度。随后 `compare_semantic_views` 按稳定拓扑序递归比较字段，首个差异抛出 `E_DIVERGE`，详情包含 `first_mismatch_shard` 和 `first_mismatch_field`。

## 9. 容量换代整体替换

容量版本递增后：

1. `initialize_capacity_slots(v2)` 建立 v2 自己的资源单位和节点租约槽位。
2. `revoke_capacity_leases(v2)` 在同一事务中撤销所有非 v2 的 ACTIVE 租约，释放资源和名额，并把对应分片置为 `FROZEN`。
3. 旧 `shard_outcomes` 与 `receipt_conclusions` 行设置 `stale=1`；摘要、水位和误差 JSON 与该行一起陈旧。
4. 当前视图只查询 `capacity_version=current AND stale=0`，v2 结果整体替换 v1，不做增量合并。
5. 快照写入新的 `snapshot_id`，记录来源快照、替换原因和新内容摘要。

## 10. SQLite 与强制终止恢复

账本默认路径：`artifacts/runtime/ledger.sqlite`。SQLite 使用 `journal_mode=WAL` 和 `synchronous=FULL`。

分片结局事务在同一个 `BEGIN IMMEDIATE` 事务中写入：

- 聚合摘要与样本快照；
- 租约最终版本/状态；
- allocation、逐个资源单位和租约槽位释放；
- 定点水位 JSON；
- 误差归因 JSON；
- 稳定拓扑游标；
- `SHARD_OUTCOME_COMMITTED` 事件。

快照表和 `meta` 保存整数 `format_version`、`spec_format_version`、`node_set_version` 与 `content_digest`。重新打开时，格式版本、节点集合版本或内容摘要任一不一致就插入新快照并整体替换当前快照，而不是合并旧负载。

SIGKILL 演示由 `edge_shard_diag.cli crash-child` 完成：子进程提交上游分片和目标租约授予后，在目标分片结局事务中执行未提交写入，随后立即 `os.kill(SIGKILL)`。父进程确认退出码为 `-9`，并终止整个子进程组以清理节点子进程。重启时：

- SQLite 自动回滚未提交事务，部分摘要不可见；
- 已提交的 ACTIVE 租约和资源槽位由 `recover_orphans` 回收；
- 对应分片标记为 `status=UNKNOWN, unknown=1, rerun_required=1`；
- 不生成所谓“被杀时退出报告”，验收 JSON 明确写为 `unavailable_by_design`；
- 调度器依据容量版本和稳定拓扑游标只重跑未完成/未知分片；
- 恢复后的语义视图与空白全量真实调度结果比较，差异抛 `E_REPLAY`。

Tkinter 界面的分片表用红色 `⚠ UNKNOWN/需重跑` 标记未知分片，右侧列表显示全部快照版本、格式版本、节点集合版本、容量/定点版本和摘要前缀。
