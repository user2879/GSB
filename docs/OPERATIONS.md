# BlockLedger 运行与复核说明

## 1. 环境

本机实际环境为 WSL Ubuntu 22.04、Python `3.11.0rc1`、NumPy `1.26.4`。基础镜像缺少 `ensurepip` 且 sudo 需要密码，因此实际使用的等价引导方式是：

```bash
python3 -m venv --without-pip .venv
. .venv/bin/activate
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python /tmp/get-pip.py
pip install -r requirements.txt
```

若系统已安装 `python3.11-venv`，可直接按任务书执行 `python3 -m venv .venv && . .venv/bin/activate && pip install numpy==1.26.4`。

当前精简 Python 运行时缺少 `libtk8.6.so`，所以 `python -m blockledger gui-check` 实际输出 `skipped`；不影响 spec、调度、SQLite 账本、恢复和独立校验。安装系统包 `python3-tk`/`tk8.6` 并提供 `DISPLAY` 后可运行 `python -m blockledger gui`。

## 2. 常用命令

```bash
# 生成随首版提供的确定性样本
python -m blockledger generate-samples

# 强校验并输出逐卷块数、快照引用、基线摘要
python -m blockledger validate --spec samples/spec.json --out artifacts/validation.json

# 确定性重放，输出每个事件的生效块集合与依据
python -m blockledger replay --spec samples/spec.json --out artifacts/replay-effects.json

# 运行配额调度、子进程校验、删除传播并提交 SQLite 权威账本
python -m blockledger run --spec samples/spec.json

# 仅依赖原始 JSON + 只读账本独立重推并逐块比较
python -m blockledger verify --spec samples/spec.json

# 失效子集重算，并与从零全量重建做逐字段一致性比较
python -m blockledger invalidate \
  --spec samples/spec.json \
  --change samples/change-write.json \
  --out artifacts/invalidation-write.json

# 恢复 SIGKILL/未提交租约；--rerun 重跑 unknown 块
python -m blockledger recover --spec samples/spec.json --rerun

# 一键本地验收
bash scripts/acceptance.sh | tee artifacts/acceptance-output.txt
```

## 3. JSON 声明

合并文件见 `samples/spec.json`，也支持目录形式：`samples/volumes.json`、`samples/snapshots.json`、`samples/events.json`、`samples/budget.json`。

- 卷：`id`/`volume_id`、`block_size`（64 到 1 MiB 之间的 2 的整数幂）、`block_count`、整数 `seed`、`baseline_version`。
- 快照：`id`/`snapshot_id`、`volume_id`、`version`、半开区间 `intervals: [[start,end)]`、按真实块号为键的 `digests`。
- 事件：严格递增 `seq`；类型支持 `write`、`snapshot_create`、`snapshot_delete`、`rollback`。
- 预算：`per_volume_rollback_concurrency`、`global_rollback_concurrency`、`max_blocks_per_rollback`、`verification_sample_permille`，另支持可选 `verification_timeout_ms`。

非法输入抛出/输出 `E_SPEC`，包含首个非法条目的 `entry_id`、`field`。退出码为 `2`。

## 4. 确定性摘要与块字节

初始块使用 NumPy `PCG64`，每块从以下材料派生独立子种子，因此与 JSON 加载顺序、批次切分和并发度无关：

```json
{
  "domain": "blockledger.initial-block.v1",
  "material": {
    "volume_id": "...",
    "master_seed": 2026112701,
    "block_index": 3,
    "baseline_version": 1,
    "block_size": 512
  }
}
```

写块使用域 `blockledger.write-block.v1`，材料包含事件 `seq`、写入种子、卷和块号。

内容摘要 preimage 为：

```text
BLOCKLEDGER-CONTENT-V1\n
|| len(UTF-8 volume_id) u64be || volume_id
|| block_index u64be
|| baseline_version u64be
|| len(content) u64be || content
```

然后取 SHA-256。快照只保存 `{block_index: metadata_bound_digest}`、块基线和块版本，不复制快照时刻的全部块字节；字节由确定性声明/事件重放或内容寻址存储解析。

## 5. 状态推进与版本

- 初始块版本为 `1`。
- 每个到达的 write 对区间内块生成确定性字节，块版本加 `1`，摘要绑定当时卷基线。
- `snapshot_create` 冻结当前块摘要、块基线和块版本，并核对 JSON 中声明的摘要。
- rollback 从 CAS/重放结果恢复目标快照摘要对应字节，每个目标块版本加 `1`，块基线回到快照捕获时的基线；随后卷全局 `baseline_version` 严格加 `1`。这样既保证恢复后摘要等于快照期望摘要，也保留卷基线推进事实。
- delete 将快照置为 tombstone，并沿 `snapshot -> rollback -> block outcome/cursor/quota history` 传播。

`artifacts/replay-effects.json` 含每个事件的 `effective_blocks` 和 `basis`。样本全量重放指纹为：

```text
9503f0247aa2a66b7513bb479bc4ed93fa5c5008cc108f9d205d5e0f947bae09
```

## 6. 抽样和子进程校验

抽样单位为 `(rollback_id, volume_id, block_index)`，规则：

```text
int(sha256("blockledger-sample-v1|<rollback>|<volume>|<block>")[:16], 16) % 1000
    < verification_sample_permille
```

- `1000` 全选，`0` 全不选。
- 未抽中：`verification_status=unchecked`，不计入通过。
- 抽中：真实启动 `.venv/bin/python -m blockledger.worker` 子进程读取块文件并计算摘要。
- 摘要不一致：批次提交 `corrupt`，CLI 输出 `E_CORRUPT`、卷、块、expected/actual，退出码 `3`。
- 失控 worker：父进程在超时后对新进程组发送 `SIGKILL`，记录 `worker_returncode=-9`、`force_killed=true`，结论为 `unknown`，必须 rerun；没有使用线程超时冒充进程终止。

## 7. 并发配额

`blockledger/scheduler.py` 只在以下条件同时满足时向 `ThreadPoolExecutor` 提交任务：

```text
active_global < global_rollback_concurrency
active_by_volume[volume_id] < per_volume_rollback_concurrency
```

样本预算为全局 `2`、每卷 `1`、单批最多 `4` 块。8 个批次的实际等待指标见 `artifacts/run-report.json` 的 `wait_metrics`；可看到同卷批次串行、第三个卷等待全局槽位。等待开始/结束使用 `time.monotonic_ns() // 1_000_000`。

## 8. SQLite 账本和会话替换

默认数据库为 `ledger/blockledger.sqlite`，sidecar 为 `ledger/session.json`。主要表：

- `volumes` / `snapshots` / `event_effects`
- `batches`：批次、块集合、等待时间、配额上限和占用
- `block_outcomes`：只追加结局历史
- `block_state_latest`：每个 rollback/volume/block 的最新结论
- `progress_cursors`：进度和各状态计数
- `quota_events`：获取、释放、恢复释放
- `commit_records`：整数 `format_version` 与提交内容摘要
- `invalidations`：失效传播步骤

`complete_batch()` 在同一个 SQLite 事务中写块结局、恢复后摘要、校验状态、配额占用/释放、进度游标、提交摘要和全账本摘要。

重新打开时比较 `format_version`、整数 `volume_set_version` 和规范化 spec 的 `content_digest`。任一不一致：

1. 用 SQLite backup API 备份旧库到 `ledger/backups/*.sqlite`；
2. 备份设为只读 `0444`；
3. 写 `*.diff.json` 差异列表；
4. 整体创建新库，不做增量合并。

## 9. 强制终止恢复和独立重放

批次开始前先在独立事务提交 `batches.status=running` 租约和块集合。若进程在批次完成事务前被 `SIGKILL`：

- 已提交租约仍为 `running`；
- 完成事务中的块结局/游标不会部分提交；
- `recover` 将租约内所有块写入 `unknown`，把批次改为 `interrupted_unknown`，写 `recovery_release` 回收配额；
- `recover --rerun` 只重跑最新状态为 `unknown` 的块。

`python -m blockledger verify` 以 SQLite read-only URI 打开账本，只读取原始 JSON 和账本，自行重放事件、抽样、推导删除失效状态，再按 `(volume_id, block_index, rollback_id)` 排序逐字段比较。首个差异输出 `E_VERIFY` 及 `volume_id/block_index/field`，退出码 `4`。

## 10. 失效子集

`blockledger/invalidation.py` 对每个种子块独立重算块时间线，包括 write、snapshot create、rollback 和 volume baseline 递增。写变更的种子集合是旧/新区间并集；删除的种子集合是快照区间。

输出包含：

- 按 `(volume_id, block_index)` 排序的 `affected_set`
- 快照期望摘要变化
- rollback 期望摘要、基线、块版本变化
- 账本键
- 每一步传播依据
- `full_rebuild_consistency.match=true`

未受影响条目不加入补丁，并与从零全量重放的 `snapshots`/`rollbacks` 规范化投影逐字段比较。
