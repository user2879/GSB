# 实际验证记录

验证时间：2026-11-27（Asia/Shanghai）。

## 环境

```text
Linux/WSL Ubuntu，Python 3.11.0rc1
.venv/bin/python -> python3 -m venv --without-pip + get-pip
NumPy 1.26.4
SQLite 3.37.2
Tk/Tkinter 8.6（doctor 导入成功；无 DISPLAY 时不创建窗口）
```

初始化命令：

```bash
python3 --version
python3 -m venv --without-pip .venv
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
.venv/bin/python /tmp/get-pip.py
.venv/bin/python -m pip install numpy==1.26.4
```

当前沙箱无免密 sudo，系统最初缺少 `libtk8.6.so`。已执行 `scripts/install-local-tk.sh` 的无 root 等价操作，将 `libtcl8.6`、`libtk8.6`、`libxft2`、`libxss1` 解包到 `.sysdeps/root`；`shard_diag.cli doctor` 会预加载这些库。标准 WSL 也可使用：

```bash
sudo apt-get update
sudo apt-get install -y python3-tk
```

## 一键验收

```bash
./scripts/demo.sh
```

最后一次实际结果：

```text
doctor: socket/threading/sqlite3/numpy/tkinter 全部 OK
validate: ok=true, nodes=3, shards=12, events=15
ports: 127.0.0.1:39101/39102/39103 available
run: completed=8, cancelled=1, invalidated_rollback=3, active_leases=0
verify: ok=true
  lease_event_count=28
  peak_resource_units=5
  unique_active_ownership=true
  journaled_events=15/15
  rollback invalid_shards=[s-v2-fast, s-v2-long, s-rb-child, s-rb-grand]
recompute priority edit:
  first_changed_seq=8
  aggregate_recompute_shards=[]
  cache_hits=12
  outcome_rows_unchanged=true
unittest: Ran 27 tests ... OK
DEMO_OK ledger=.runtime/demo.sqlite
```

账本根内容摘要（另一次完整运行的 SQLite 文件示例）：

```text
88898056d8312df2a6e5430edb0cb255960310e17a326489ec39f43ed83106d8
```

根摘要包含 `updated_at` 和真实单调时钟毫秒，因此每次运行的根摘要会不同；应逐字节复核的聚合摘要和样本字节是确定性的。

## 逐字节确定性复核

对样例 `n-beta / s-independent`，分别用批次大小 `1,7,12,24,99` 调用 NumPy 分组聚合，得到同一个摘要：

```text
aggregate_digest=390c5e1cd674de779cb10d8819f641c61c381e2d64ee395d2f04cb7a3a53b3e6
sample_raw_sha256=73507324f130352d773963b6c28a979439dc91ab5e3f72404bc9aba3c5884854
distinct_digests_across_batch_splits=1
```

SQLite 中该分片记录的 `aggregate_digest` 与独立重算一致；`verify` 命令会对所有 `completed` 分片执行相同检查。

## 已覆盖的异常码

- `E_SPEC`：重复节点 ID、非 `127.0.0.1` host、重复端口、布尔冒充整数、缺失依赖、DAG 环、事件引用/顺序错误。测试：`tests/test_spec_validation.py`。
- `E_PORT`：真实占用端口后 `ensure_port_available` 报告端口和节点对象。测试：`tests/test_protocol.py::test_port_occupied_code`。
- `E_TIMEOUT`：假 TCP 服务接受连接但不回应，60ms 真实 socket 超时。测试：`tests/test_protocol.py::test_timeout_code`。
- `E_FIELD`：握手缺少 `node_id` 或聚合回执缺少 `partial`。测试：`tests/test_protocol.py::test_missing_handshake_field_code`、`test_missing_result_field_code`。
- `E_LEASE`：节点侧错误 `expected_version=999` 被拒，随后旧版本 `1 -> 2` 成功，证明失败续约没有推进节点代数；主注册表也验证过期续约和 stale token。测试：`tests/test_scheduler_leases.py`。
- `E_VERIFY`：篡改 `.runtime`/临时账本中 `old` 分片的 `aggregate_digest` 后，校验定位到 `object_id=old`、`field=aggregate_digest`。测试：`tests/test_ledger_rollback_verify.py`。

## 并行、容量与租约

`tests/test_scheduler_leases.py` 使用真实 `NodeManager` 和 `ThreadPoolExecutor` 验证：

- 单节点并发配额为 1 时，第二个分片先得到 `node_concurrency_quota` 排队依据，首个结束后放行。
- 全局预算不足时得到 `global_budget` 排队依据。
- 每节点租约上限为 1 时得到 `node_lease_cap` 排队依据。
- 并发配额为 2、两个分片各注入 250ms 节点处理时，账本中的 started/finished 区间重叠至少 100ms，且使用两个不同 worker 线程。
- 等待开始/结束使用 `time.monotonic_ns()` 转换的毫秒值。
- 节点每个批次回执带 `duration_monotonic_ms`，由节点侧 `time.monotonic_ns()` 实测。
- TTL 40ms 后真实 sleep 70ms，`reap_expired()` 才回收；没有可注入的虚假时钟。

## 回滚唯一归属与恢复

`tests/test_ledger_rollback_verify.py` 验证：

- 回滚输出稳定失效集 `['new', 'child']` 及每步传播依据。
- 回滚前后未失效 `old` 分片的整行账本字典完全相等。
- 租约状态只能从 `active` CAS 到 `expired`、`revoked_rollback`、`released`、`cancelled` 或 `failed` 一次；同一资源不会同时进入过期回收和回滚撤销。
- 声明摘要变化后重新打开快照会 `replaced=true`，旧 meta 标记不会合并。
- 有 `grant` 和事件日志、但没有原子结局的 `ghost` 分片被 `recover_incomplete_shards` 标记为 `unknown`，动作是 `rerun_required`。

## 进程/端口清理检查

演示结束后执行：

```bash
pgrep -af 'shard_diag.node_worker'
ss -ltnp | rg '3910[123]'
```

没有遗留节点工作进程或监听端口。`tests/test_protocol.py::test_real_subprocess_aggregate_force_kill_releases_port` 还会在长聚合请求进行中强杀整个节点进程组，确认端口可立即重新绑定，并能重新 `start_node()` 后响应 ping。

## 增量编辑

`examples/events_priority_edit.json` 只把 seq 8 的优先级从 90 改为 123。`recompute` 报告首个变化序号为 8，聚合重算集合为空，12 条已有结局全部作为缓存命中，并且增量过程前后 `shard_outcomes.content_digest` 映射完全不变。

若编辑的是 dispatch、quota_upgrade 或 rollback，`impact.aggregate_recompute_shards` 会包含节点/配额输入发生变化的分片；依赖传播集合和脏水位作用域由 `shard_diag/impact.py`、`shard_diag/incremental.py` 输出。

## 未做事项

- 没有使用 Docker 或任何外部服务/账号。
- 没有访问非 `127.0.0.1` 目标。
- 未创建 Tk 图形窗口；当前 WSL 会话无 `DISPLAY`，但 Tkinter 导入自检和全部 CLI/账本/重放功能可用。
