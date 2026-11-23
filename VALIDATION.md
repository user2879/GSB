# 实际验证记录

验证日期：2026-11-24  
运行环境：Python 3.11.0rc1（代码使用 Python 3.10+ 语法），Linux 工作区；实现按 Windows 本地子进程、回环套接字和 SQLite 语义设计。  
依赖准备：`python -m pip install numpy`（核心实现为 Python 标准库）。

## 1. 全量测试

```text
$ python -m unittest discover -v
...
Ran 26 tests in 8.840s

OK
```

退出码为 `0`。覆盖规格拒绝、TCP 错误、唯一仲裁、排队、竞态回收、强杀恢复、快照替换、独立重放和节点收敛。

## 2. 本地样本端到端结果

命令：

```bat
python -m gsbcache sample --out runtime\final-demo
python -m gsbcache run --spec runtime\final-demo\spec.json --work runtime\final-demo\work --lease-ms 420
python -m gsbcache verify --spec runtime\final-demo\spec.json --ledger runtime\final-demo\work\ledger.sqlite3
python -m gsbcache converge --spec runtime\final-demo\spec.json --ledger runtime\final-demo\work\ledger.sqlite3 --nodes-root runtime\final-demo\work\nodes
```

实际结果：

```text
RUN True duration_ms=887
STAGE prepare quota=2 workers=2 observed_max=2 duration_ms=24
  src:main    node-a queue_ms=0   total_queue_ms=1   exec_ms=8  event_seq=1
  src:util    node-b queue_ms=6   total_queue_ms=6   exec_ms=11 event_seq=2
STAGE compile quota=2 workers=2 observed_max=1 duration_ms=429
  app:bin     node-a queue_ms=409 total_queue_ms=410 qpos=1 exec_ms=8 event_seq=5
STAGE test quota=1 workers=1 observed_max=1 duration_ms=14
  test:report node-b queue_ms=0   total_queue_ms=0   exec_ms=8  event_seq=6
EXPIRY {"active_leases": 0, "expired": true, "wait_ms": 420}
VERIFY True events_replayed=8
CONVERGE True checked_artifacts=4
```

`app:bin` 的排队依据实际为：

```text
node-a:lease_limit_exhausted:limit=1:active=1:pending=0;
node-b:capacity_slots_exhausted:slots=1:used=1:pending=0;
node-c:digest_mismatch:expected=0b40106f...:actual=f25eb820...
```

即样本同时验证了每节点租约上限排队、容量槽位排队和 seed 摘要不匹配否决。观测到的全局在途最大值为 2（上限 3），node-a/node-b 在途最大值分别为 1（配额分别 1/2），没有越过配额。

事件账本：

```text
1 lease_granted   node-a ["src:main"]
2 lease_granted   node-b ["src:util"]
3 lease_expired   node-a ["src:main"]     capacity_slots_released=1 quota_units_released=1
4 lease_expired   node-b ["src:util"]     capacity_slots_released=1 quota_units_released=1
5 lease_granted   node-a ["app:bin"]
6 lease_granted   node-b ["test:report"]
7 lease_expired   node-a ["app:bin"]      capacity_slots_released=1 quota_units_released=1
8 lease_expired   node-b ["test:report"]  capacity_slots_released=1 quota_units_released=1
```

最终节点余量：

```text
node-a: capacity 2/2, active_leases 0, quota 1/1
node-b: capacity 1/1, active_leases 0, quota 2/2
node-c: capacity 4/4, active_leases 0, quota 2/2
```

会话为 `state=completed, exit_report=1`。

独立重放输出：

```json
{
  "ok": true,
  "code": "OK",
  "events_replayed": 8,
  "conclusion": "ledger replay matches independently derived digests, ownership and custody"
}
```

收敛输出：

```json
{
  "ok": true,
  "checked_artifacts": 4,
  "conclusion": "all persisted node replicas converge with ledger digests"
}
```

## 3. 失败注入验证

- `E_PORT`：占用 `node-a` 声明端口后启动集群，返回 `E_PORT`，包含 `node_id`、`host`、`port`。
- `E_HANDSHAKE`：本地假服务器接受连接但不回应，客户端以 120ms 超时返回 `E_HANDSHAKE`。
- `E_REPLY`：假服务器接受租约请求但省略 `lease_id/granted_keys/expires_at_ms/digests/reusable_keys`，客户端返回 `E_REPLY` 并列出缺失字段。
- 到期/配额竞态：两个线程同时调用 `expiry` 与 `quota_recycle`；实际只有一个 `closed=true`，另一个为 `already_closed=true`，账本只有一条释放事件，节点容量与配额均恢复一次。
- 强杀恢复：在节点授予后、账本提交前注入 5 秒暂停并 `kill -9` 控制器。重启后观察到 `pending` 尝试变为 `unknown`、`requires_reapply=true`、会话 `aborted` 且 `exit_report=0`，恢复结果 `exit_report=null`；随后重跑成功并通过独立重放。
- 快照替换：篡改账本 `meta.content_digest` 后重新打开返回 `E_LEDGER` 和差异列表；`ledger-replace` 以 `whole_snapshot_replace` 模式恢复，`incremental_merge=false`，恢复后 `verify` 返回 OK。
- 独立重放差异：重写事件摘要并同步重算事件链与账本摘要后，`verify` 返回 `E_VERIFY`，定位到首个 `artifact_key`、`node_id`、`field=digests` 和 `event_seq`。
- 副本收敛差异：强杀并重启节点前篡改其持久化制品字节，节点重启后从文件重算得到新摘要且可复用性变为 false；`converge` 返回 `E_CONVERGE`，报告偏差节点 `[node-a]` 和制品键集合 `{src:main}`。
