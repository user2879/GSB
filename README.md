# 云原生构建流水线制品缓存租约仲裁组件

这是一个面向 Windows 本地环境、从零实现的可执行参考实现。它把制品缓存复用建模为可独占、可到期、可持久恢复的租约资源，并同时交付：

- 强校验的节点/流水线声明加载器；
- 确定性制品内容与摘要生成器；
- 真实回环 TCP JSON-lines 缓存节点；
- 多节点唯一仲裁、容量/租约配额排队、线程池调度；
- SQLite 仲裁账本、事件哈希链、整库快照替换与强杀恢复；
- 独立重放验证器 `verify` 和节点副本收敛工具 `converge`；
- 26 个本地 `unittest` 验收测试。

实现只使用 Python 标准库；按照题目准备步骤执行 `python -m pip install numpy` 即可，核心运行不依赖外部服务、账号、Docker 或网络。

## 环境与快速验收

```bat
python --version
python -m pip install numpy

python -m gsbcache sample --out runtime\demo
python -m gsbcache validate --spec runtime\demo\spec.json
python -m gsbcache run --spec runtime\demo\spec.json --work runtime\demo\work --lease-ms 420 --result-out runtime\demo\result.json
python -m gsbcache verify --spec runtime\demo\spec.json --ledger runtime\demo\work\ledger.sqlite3
python -m gsbcache converge --spec runtime\demo\spec.json --ledger runtime\demo\work\ledger.sqlite3 --nodes-root runtime\demo\work\nodes

python -m unittest discover -v
```

`sample` 会自动寻找三个空闲回环端口写入 `spec.json`，并在 `published/` 下物化清单声明的制品文件。`run` 会启动真实子进程节点、创建 SQLite 账本和节点数据目录，结束后自动执行独立重放与收敛校验。

实际运行结论见 [VALIDATION.md](VALIDATION.md)。

## 规格文件

单个 JSON 同时声明节点和流水线：

```json
{
  "format_version": 1,
  "nodes": [
    {
      "node_id": "node-a",
      "host": "127.0.0.1",
      "port": 43101,
      "capacity_slots": 2,
      "lease_limit": 1,
      "seed": 17
    }
  ],
  "pipeline": {
    "pipeline_id": "demo",
    "global_concurrency": 3,
    "stages": [
      {
        "stage_id": "prepare",
        "concurrency": 2,
        "artifacts": [
          {
            "key": "src:main",
            "expected_digest": "<64-lowercase-hex>",
            "refs": []
          }
        ]
      }
    ]
  }
}
```

强校验包括：

- `format_version` 为整数 `1`；
- 节点标识唯一；制品键在整条流水线内唯一；阶段标识唯一；
- `host` 必须是字面量 `127.0.0.1`；端口必须是 `1..65535` 的整数且全局不重复；
- `capacity_slots`、`lease_limit`、阶段 `concurrency`、全局 `global_concurrency` 均为正整数；
- `seed` 为整数；
- `expected_digest` 为 64 位小写十六进制 SHA-256；
- 每个 `refs` 目标存在，引用图无环；
- 校验按节点文档顺序、再按阶段/制品文档顺序执行，发现首个错误立即抛出 `E_SPEC`，并携带 `node_id` 或 `artifact_key`。

流水线阶段会按跨阶段制品引用进行拓扑排序后执行。

## 确定性内容与可复用性

节点不使用随机数生成内容。显式规则为：

```text
content = UTF-8 canonical JSON:
  {"artifact_key": <key>, "content_version": 1, "node_seed": <seed>}
        sort_keys=True, separators=(",", ":"), ensure_ascii=True
digest  = SHA-256(content)
```

节点标识不进入内容，因此相同 seed 的不同节点会生成逐位相同的副本，可参与同一制品键的竞争；不同 seed 会产生不同摘要。

节点把内容写入：

```text
<nodes-root>/<node_id>/artifacts/artifact-<sha256(artifact_key)>.bin
```

重启时，如果文件存在则直接读取并重新计算摘要，不静默重写。制品可复用必须同时满足：

1. 节点实际摘要等于清单 `expected_digest`；
2. `refs` 闭包内每个引用制品在该节点的摘要也等于其清单声明摘要。

失败输出包含 `node_id`、`expected_digest`、`actual_digest`、`reason`，引用失败还包含 `bad_references`。

## TCP JSON-lines 协议

每个请求/应答是一行紧凑 JSON。租约请求示例：

```json
{"type":"lease","protocol":1,"request_id":"demo:compile:app:bin","sequence":3,"artifact_keys":["app:bin"],"lease_ms":420}
```

节点成功应答包含：

```json
{
  "ok": true,
  "type": "lease_granted",
  "protocol": 1,
  "lease_id": "lease_...",
  "request_id": "demo:compile:app:bin",
  "sequence": 3,
  "node_id": "node-a",
  "granted_key": "app:bin",
  "granted_keys": ["app:bin"],
  "expires_at_ms": 1795459229404,
  "digests": {"app:bin": "<sha256>"},
  "reusable_keys": ["app:bin"]
}
```

错误分类稳定为：

| 代码 | 场景 | 进程退出码 |
|---|---|---|
| `E_SPEC` | 声明/引用/端口/整数等规格错误 | 2 |
| `E_PORT` | 节点子进程绑定回环端口失败 | 3 |
| `E_HANDSHAKE` | 节点未就绪或握手超时/失败 | 4 |
| `E_REPLY` | 应答缺字段、超时、连接中断 | 5 |
| `E_VERIFY` | 独立重放与账本不一致 | 6 |
| `E_CONVERGE` | 节点落盘副本摘要与账本不一致 | 7 |
| `E_LEDGER` | 账本版本/摘要/快照完整性错误 | 8 |
| `E_ARB` | 制品键冲突、无候选或仲裁队列超时 | 9 |

节点使用逐连接线程处理协议行；控制器使用 `subprocess.Popen` 启动 `python -m gsbcache.node`，Windows 下设置 `CREATE_NO_WINDOW`，并传递父进程号。父进程被强杀后，节点会检测父进程消失并自行退出，避免遗留端口。

## 仲裁与调度

候选节点必须同时满足：摘要匹配、引用闭包可复用、剩余容量槽位大于 0、剩余租约配额大于 0。可服务候选按以下稳定键排序：

```text
摘要匹配优先（不匹配者直接否决）
→ 剩余容量槽位降序
→ node_id 字典序升序
```

所有候选（包括被否决者）会写入请求结果和 `lease_granted` 事件，包含实际/期望摘要、剩余容量、剩余配额、排名与否决理由。

配额耗尽时：

- 控制器在条件队列中等待，并输出 `queue_position`、`queue_reason`、容量等待时间、线程池/并发门等待时间；
- 到期回收后唤醒等待者重新仲裁；
- 同一制品键已被活跃租约或另一个请求持有时立即返回 `E_ARB`，不会同时授予两个节点。

流水线调度真实使用 `ThreadPoolExecutor`。阶段线程池与阶段信号量执行阶段并发配额；全局信号量执行全局在途请求上限；节点信号量执行每节点在途请求配额。每个请求输出：

- `stage_queue_ms`：等待阶段工作线程；
- `queue_ms` / `total_queue_ms`：容量/配额及并发门等待；
- `execution_ms`：真实 socket 请求执行时间；
- 单调时钟起止毫秒和观测到的在途数。

## 到期与唯一归属

控制器为每个租约维护 `deadline_mono_ns`，到期回收线程用 `time.monotonic_ns()` 真实计时，不使用人工 `sleep` 伪造事件。到期或配额回收都在一个 SQLite 事务中：

1. 条件更新 `leases.state='active' → 'closed'`；只有更新成功的路径能继续；
2. 同时释放容量槽位与租约配额；
3. 写入唯一释放事件、`custody_token=<lease_id>`、释放原因和事件序号；
4. 更新节点的 `capacity_used/capacity_remaining/active_leases/quota_remaining`、进度游标和内容摘要。

因此到期回收与配额回收竞态时只有一个赢家；不会重复释放同一份容量权益，也不会出现两个活跃租约持有同一制品键。

## SQLite 账本与恢复

账本表包含 `meta`、`nodes`、`leases`、`events`、`attempts`、`sessions`。关键性质：

- 授予、续约、到期回收、配额回收均在同一事务中写入事件、节点余量、租约集合和游标；
- 事件序号连续，并使用 `prev_hash + canonical_event` 形成 SHA-256 事件链；
- `meta.content_digest` 覆盖元数据、节点、租约、尝试、会话和全部事件；
- 账本首次写入整数 `format_version=1` 和清单摘要。

恢复语义：

- 网络请求前先提交 `attempts.state='pending'`；
- 节点授予后，只有在账本授予事务提交后才算完整授予；
- 强杀后 SQLite 自动丢弃未提交事务；重新打开时，已提交的 `pending` 尝试被标为 `unknown`，并追加 `attempt_unknown` 事件要求重新申请；
- 上一个会话遗留的活跃租约被标为 `lease_unknown` 并释放，不继续假定可用；
- `sessions.state='aborted', exit_report=0`，恢复结果 `exit_report=null`，不会声称能生成退出报告。

快照工具：

```bat
python -m gsbcache ledger-export --ledger ... --spec ... --out snapshot.json
python -m gsbcache ledger-replace --ledger ... --snapshot snapshot.json --spec ...
```

`ledger-replace` 校验快照格式版本、内容摘要和事件链后，使用临时 SQLite 文件重建全部状态，再一次性替换目标文件；绝不增量合并，并输出稳定排序的版本、摘要、游标和各表计数差异。

## 独立验证与收敛

`verify` 只读取账本和原始声明，不读取节点运行状态。它独立重做：

- 内容摘要生成；
- 引用闭包可复用性；
- 候选节点排序和唯一赢家推导；
- 容量槽位、租约配额、单键唯一性；
- 授予/续约/到期/配额回收归属；
- 最终节点余量与活跃租约集合。

首个差异以 `E_VERIFY` 返回，并定位 `artifact_key`、`node_id`、`field`、`expected`、`actual`、`event_seq`。

`converge` 读取每个节点目录中持久化的制品文件，重新计算 SHA-256，并与账本中该节点/制品的摘要逐项比较。偏差以 `E_CONVERGE` 返回，包含：

- `deviating_nodes`：偏差节点列表；
- `deviating_artifact_keys`：每个节点对应的制品键集合；
- `deviations`：逐项路径、原因、期望与实际摘要。

## 测试覆盖

`python -m unittest discover -v` 覆盖 26 个场景，包括：

- 所有 `E_SPEC` 强校验及首个错误定位；
- 确定性摘要、seed 不匹配、引用闭包不匹配；
- `E_PORT`、`E_HANDSHAKE`、`E_REPLY`；
- 多节点唯一仲裁、同键竞争、到期后重新申请；
- 租约上限与容量槽位排队、线程池/全局/节点并发上限；
- 到期回收与配额回收竞态只有一个赢家；
- 强杀后 `unknown/requires_reapply` 与无退出报告；
- 快照摘要篡改检测和整库替换；
- 独立重放 `E_VERIFY` 定位；
- 节点重启后持久内容摘要偏差 `E_CONVERGE`。
