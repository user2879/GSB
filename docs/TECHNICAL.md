# 推荐特征缓存诊断工作台技术说明

## 1. 交付范围

本组件是一个本地、确定性、无外部服务的跨节点缓存诊断工作台：

- 从 JSON 配置加载节点、容量水印策略、特征构建清单和事件流。
- 为每个有效节点真实启动一个 `127.0.0.1` JSON Lines TCP 子进程。
- 支持 `write`、`read`、`release`、`quota_change` 四类协议请求。
- 计算定点容量水印、权威副本、租约续期/过期、牺牲者驱逐、预占/拒绝、预算扣减和内容指纹缓存。
- 阶段化写入临时目录，并通过不可变快照目录加 `current.json` 指针完成整体提交。
- 输出 JSON/Markdown 诊断报告，便于逐字段复核。

核心代码位于 `workbench/`，端到端测试位于 `tests/test_workbench.py`。

## 2. 快速运行

```bash
python -m pip install -r requirements.txt
python -m pytest tests/test_workbench.py -v

python -m workbench.cli init-sample --dir examples
python -m workbench.cli validate \
  --config examples/sample-config.json \
  --events examples/events.jsonl

python -m workbench.cli run \
  --config examples/sample-config.json \
  --events examples/events.jsonl \
  --out artifacts/demo \
  --parallel \
  --protocol-self-check
```

非法输入样例：

```bash
python -m workbench.cli validate --config examples/invalid-input.json
# 退出码为 2；JSON 中分别包含 E_NODE_INVALID、E_MANIFEST_INVALID、E_EVENT_INVALID
```

Windows PowerShell 可使用：

```powershell
py -3.11 -m pip install numpy pytest
py -3.11 -m workbench.cli init-sample --dir examples
py -3.11 -m workbench.cli run --config examples/sample-config.json --events examples/events.jsonl --out artifacts\demo
```

`init-sample` 会向操作系统申请空闲回环端口，因此不需要管理员权限，也避免固定高位端口冲突。

## 3. 输入模型

### 3.1 节点

```json
{
  "node_id": "n-water",
  "port": 43183,
  "capacity_slots": 10,
  "lease_limit": 8,
  "seed": 3303,
  "capacity_version": 1,
  "quota_version": 1
}
```

校验规则：

- `node_id` 非空且唯一。
- `port` 为 `1..65535` 整数且唯一；服务只绑定 `127.0.0.1`。
- `capacity_slots`、`lease_limit` 为正整数。
- `seed` 为整数，用于确定性单调耗时序列。
- `capacity_version` 为正整数，且必须能在 `capacity_policies` 中找到。

### 3.2 容量策略

```json
{
  "capacity_policies": [
    {"capacity_version": 1, "soft_permille": 600, "hard_permille": 800},
    {"capacity_version": 2, "soft_permille": 300, "hard_permille": 500}
  ]
}
```

策略版本必须严格递增。事件 `POLICY_CHANGED` 中的新版本也必须大于节点当前版本。

### 3.3 特征清单

每个阶段包含：

- `stage_id`
- `artifact_key`
- `needs`：上游制品键
- `upstream_stage_ids`：可选上游阶段
- `concurrency_quota`：正整数并发预算

校验器会建立制品生产者索引和阶段图，检查上游是否存在，并使用 DFS 发现环。

### 3.4 事件类型

允许的事件枚举为：

- `APPLY`
- `WRITE`
- `READ`
- `RELEASE`
- `LEASE_RENEW`
- `TICK`
- `QUOTA_CHANGED`
- `POLICY_CHANGED`
- `RUN_EVICTION`

`seq` 必须为正整数且严格递增。非法事件进入 `failure_list`，不会中断其他事件、节点或清单对象的独立加载。

## 4. JSON Lines 节点协议

每个连接发送一行 JSON 请求并读取一行 JSON 响应。

### 写入

```json
{
  "request_id": "write-seq00001-n-water-block:01",
  "type": "write",
  "key": "block:01",
  "slots": 1,
  "priority": 5,
  "ttl_ms": 100000,
  "now_ms": 3000,
  "stage_id": "rank_builder",
  "content": {"weights": [1, 2]}
}
```

成功响应包含：

- `content_digest`：对字符串按 UTF-8、对 JSON 值按规范化紧凑 JSON 计算的 `sha256:` 摘要。
- `quota_version`、`capacity_version`
- `elapsed_ms`、`clock_source`
- `result.overwritten` 与 `result.overwritten_digest`
- `lease_id`、`issued_ms`、`expires_ms`

新键必须同时满足物理槽位、硬水印和节点租约上限。超过租约上限返回 `E_LEASE_LIMIT_EXCEEDED`；超过物理槽位或硬水印投影返回 `E_CAPACITY_EXCEEDED`。

### 读取与续期

读取请求可携带 `renew_ttl_ms`。节点仅在 `now_ms < expires_ms` 时续期；到期时刻及之后的续期返回 `renewed=false, renewal_reason=lease_expired`。

为保持“四类请求”边界，续期复用 `read` 的可选续期字段；到期回收由引擎在 `TICK` 时发送真实 `release`。

### 释放

可按 `key` 或 `lease_id` 释放，返回实际释放槽位和摘要。

### 配额/策略变更

`QUOTA_CHANGED` 和 `POLICY_CHANGED` 都映射到节点的 `quota_change` 协议类型，可更新：

- `quota_version`
- `lease_limit`
- `capacity_version`

### 协议错误

非法 JSON、未知请求类型、字段类型错误、未知容量版本或单调时钟回退均返回：

```json
{"ok": false, "error": {"code": "E_PROTOCOL_VIOLATION"}}
```

CLI 的 `--protocol-self-check` 会故意发送畸形 JSON 和未知类型，并把两条 `E_PROTOCOL_VIOLATION` 写入报告与失败清单。

## 5. 确定性时钟

默认 `clock_mode="seeded"`。每个节点使用声明的整数种子初始化确定性 NumPy `RandomState`，生成稳定的请求耗时抖动；租约签发/到期使用事件中单调不减的 `now_ms`。因此同一最终事件集合在全量、增量续跑、串行和并行扇出下得到逐字段相同的报告。

如需观察真实墙钟耗时，可设置顶层 `"clock_mode": "system"`；该模式使用 `time.monotonic_ns()`，但耗时字段不再承诺跨进程重放逐字段一致。裁决和缓存键仍主要依赖内容指纹与版本，不使用随机 UUID 或 PID。

## 6. 定点水印与并列规则

水印比例单位为千分比，定点分母为 `1000`：

```text
limit = round_half_up(capacity_slots * permille / 1000)
```

- 半值向上取整，例如 3 槽 × 500‰ = 1.5 → 2。
- 占用 `>= soft_limit` 进入 `soft_warning`。
- 占用 `>= hard_limit` 进入 `hard`。
- 新申请的投影占用 `> hard_limit` 时拒绝并记录 `E_CAPACITY_EXCEEDED`；等于硬上限允许落盘并立即显示 hard，之后的申请会被拒绝。

固定决断元数据会写入报告和快照清单：

- 牺牲者：`priority` 降序 → `expires_ms` 升序 → `artifact_key` 字典序升序。
- 权威副本：`capacity_version` 降序 → `elapsed_ms` 升序 → 摘要升序（有摘要优先于无摘要）→ `node_id` 升序。
- 待处理申请：`artifact_key` 升序 → 到达 `seq` 升序 → `application_id` 升序。
- 缓存账本：`node_id` → `artifact_key` → `stage` → `scope_id` 排序。

## 7. 权威副本与覆盖引用

同一 `artifact_key` 在多个节点的当前驻留记录按上述规则裁决。账本保存：

- 胜者节点和请求序号
- 全部候选的排序元组
- 容量版本、耗时、摘要
- 每次同节点覆盖的旧摘要、新摘要、事件序号和租约 ID

样例中 `feat:user_profile` 同时写入 `n-core-a`（容量版本 1）和 `n-core-b`（容量版本 2），即使 `n-core-a` 发生覆盖，权威副本仍由更高容量版本的 `n-core-b` 获得。

## 8. 预占与真实驱逐对齐

`POLICY_CHANGED` 更新节点版本/租约上限后，引擎使用与真实回收相同的牺牲者排序生成强制驱逐计划：

1. 若 `active_leases > lease_limit`，继续选择牺牲者。
2. 若 `occupancy >= hard_limit`，继续选择牺牲者。
3. 得到计划驱逐键序列和驱逐后占用/租约数。
4. 按制品键稳定顺序模拟待处理申请；满足硬水印、物理槽位、租约上限和阶段预算则 `reserved`，否则 `rejected`。
5. `RUN_EVICTION` 按计划逐条发送真实 `release`，再按预占顺序发送真实 `write`。
6. `reconciliations` 比较 `planned_eviction_keys` 与 `actual_eviction_keys`，并记录 `matched=true/false`。

样例 `n-water` 从容量版本 1（软 6/硬 8）切到版本 2（软 3/硬 5），同时租约上限从 8 降到 6。计划驱逐序列固定为：

```text
block:04, block:02, block:01, block:03
```

随后 `af:pending:a` 被预占并准入，`af:pending:b` 因硬水印被拒绝；下一次 `block:overflow` 写入触发 `E_CAPACITY_EXCEEDED`。

## 9. 指纹缓存与精确失效

每个缓存键形式为：

```text
ck_<sha256(stage | upstream_fingerprint | quota_version | capacity_version)[:24]>
```

覆盖阶段：

- `resident_summary`
- `watermark`
- `eviction_plan`
- `authority`
- `pending_decision`

`cache_ledger` 为每个作用域保存当前 `cache_key`、上游指纹、配额/容量版本、`hit_count`、`recompute_count` 和最后状态。单次 `QUOTA_CHANGED` 只改变目标节点的 `quota_version`，因此目标节点相关键重算；未变更节点的输入指纹和版本均不变，直接返回缓存产品并记录 `hit`。

## 10. 一致性保证

所有易变状态都由同一个有序事件折叠得到：

- 事件序号严格排序。
- 跨节点并发扇出使用线程池等待全部结果，然后按节点 ID 排序更新状态。
- 节点子进程内部用锁保护驻留表和随机数序列。
- 不把 PID、墙钟时间或临时目录名放入规范报告。
- 缓存评估次数也由事件数决定；增量续跑从已完成游标继续，不额外插入评估。

测试会对样例流的每个分割点比较全量与增量报告，并比较串行与并行报告；另外使用确定性随机事件序列复核相同性质。

## 11. 快照、恢复与版本识别

输出目录结构：

```text
artifacts/demo/
  current.json
  recovery.json
  logs/<node_id>.log
  snapshots/g0001-c000040-<manifest-sha12>/
    01_load_validation.json
    02_capacity_watermarks.json
    03_leases_evictions.json
    04_ledgers.json
    resident_snapshot.json
    report.json
    report.md
    snapshot_manifest.json
```

提交过程：

1. 在输出目录下创建隐藏临时目录。
2. 写入全部阶段产物并计算 SHA-256。
3. 写入 `snapshot_manifest.json`，包含 `global_version`、`config_hash`、`quota_version`/逐节点配额版本、`event_cursor`、事件前缀指纹和 `phase_versions`。
4. 将临时目录原子替换为不可变快照目录。
5. 写 `current.json.tmp` 后 `os.replace` 为 `current.json` 指针。

启动时：

- `global_version` 与 `config_hash` 相同且事件前缀指纹相同：判定可续跑；CLI 会重放游标前事件以重建真实子进程状态，再处理后续事件。
- 配置哈希变化：`global_version += 1`，记录 `config_hash_changed`。
- 事件前缀变化：为避免错误续跑，同样提升全局版本并记录 `event_prefix_changed`。
- `--fresh`：强制开启新全局版本。

## 12. 进程生命周期

- 父进程使用 `subprocess.Popen` 启动 `python -m workbench.node_server`。
- `NodeCluster.close()` 在 POSIX 上调用 `terminate()`，超时后 `kill()`；在 Windows 上使用 `taskkill /PID <pid> /T /F`。
- `NodeCluster` 注册 `atexit` 清理。
- 节点携带 `--parent-pid` 并运行守护线程；父进程异常退出后，POSIX 用 `kill(pid, 0)`、Windows 用 `OpenProcess/WaitForSingleObject` 检测父进程并自行 `_exit`。

因此正常退出、异常折叠和父进程退出路径都会真实回收子进程，而不是只关闭 TCP 套接字。

## 13. 已验证结果

实际命令与输出摘要见 `artifacts/verification/pytest.txt` 和 `artifacts/verification/demo-run.json`。当前记录为 WSL2/Linux、Python 3.11.0rc1、NumPy 1.26.4、pytest 8.2.0；Windows 进程路径已按跨平台 API 实现，但本次未在原生 Windows runner 上执行。
