# 本地多集群配置分发状态演进与恢复一致性验证管线

这是一个只面向本机回环网络的确定性分发管线。它从 JSON 清单读取集群、任务和配置修订链，为每个集群启动独立模拟服务子进程，在全局资源、集群配额和任务并发三重约束下并行推进分片，并将结局、聚合摘要、资源消耗和游标写入 SQLite。修订演进时只重算失效子集；旧世代整体失效；强制终止后只接受完整提交的记录，并通过真实模拟服务重放校验摘要。

## 环境与启动

- Python 3.11+
- NumPy（唯一第三方依赖，用于分组聚合与确定性样本）
- 标准库：`sqlite3`、`socket`、`subprocess`、`threading`、`tkinter`、`time` 等
- 不使用 Docker 或其他容器

```powershell
python -m pip install -r requirements.txt
python scripts/generate_samples.py --output-dir samples
python -m config_pipeline validate samples/manifest.json
python -m config_pipeline run samples/manifest.json --db artifacts/demo.db
```

Windows 下模拟服务通过 `python -m config_pipeline.mock_service` 子进程启动，并设置 `CREATE_NO_WINDOW`/`SO_EXCLUSIVEADDRUSE`（可用时）。服务会监听父进程 PID，父进程被强杀后自动退出并释放端口。端口 1–1023 在 Windows 上通常需要管理员权限。

## 清单格式

规范字段如下；也兼容列出的常见别名。

```json
{
  "global_resource_budget": 4,
  "clusters": [
    {
      "id": "alpha",
      "host": "127.0.0.1",
      "port": 43101,
      "nodes": ["alpha-1", "alpha-2"],
      "concurrency": 2,
      "seed": 101
    }
  ],
  "tasks": [
    {
      "id": "rollout",
      "config_keys": ["capacity", "scaled"],
      "target_clusters": ["alpha"],
      "shards": 4,
      "resource_units": 1,
      "concurrency_limit": 2
    }
  ],
  "revisions": [
    {
      "version": 1,
      "parent": null,
      "values": {"capacity": 10, "threshold": 3},
      "derived": {
        "scaled": {"op": "sum", "inputs": ["capacity", "threshold"]}
      }
    },
    {
      "version": 2,
      "parent": 1,
      "values": {"capacity": 14}
    }
  ]
}
```

字段规则：

- 集群：`id`、`host`、`port`、`nodes`、`concurrency`，可选 `seed`。`host` 缺省为 `127.0.0.1`，显式提供时必须是该字面量；端口必须为 1–65535 的整数且全局唯一；集群标识和节点内标识唯一；并发配额必须为正整数。
- 任务：`id`、`config_keys`、`target_clusters`、`shards`、`resource_units`，可选 `concurrency_limit`。分片数和每分片资源单位为非负整数；任务并发上限缺省为 `max(1, shards)`；目标集群必须存在。
- 修订：`version`、`parent`、`values`、`derived`。数组内版本严格递增，首修订父版本为 `null`，后续父修订必须更早存在，因此不可能成环。`values` 是相对父修订的增量并继承未覆盖值；`derived` 可继承或覆盖派生规则。支持 `sum`、`min`、`max`、`product`、`sub`；派生图必须无环且所有输入可解析。
- 任一每分片资源单位不得超过全局预算，否则该分片永远不可能被调度，按 `E_MANIFEST` 拒绝。
- 常用别名：`resource_budget`/`global_budget`/`budget`、`loopback_port`、`targets`/`target_cluster_ids`、`shard_count`、`resources_per_shard`、`task_concurrency`/`max_concurrency`、`parent_version`、`config_values`、`operation`/`from` 等。
- JSON 重复对象键也会被拒绝。所有错误定位到首个违规字段，例如 `clusters[1].port` 或 `tasks[0].resource_units`。

## CLI

```text
python -m config_pipeline validate MANIFEST
python -m config_pipeline compile MANIFEST [--output PLAN.json]
python -m config_pipeline run MANIFEST --db STATE.db [--snapshot SNAPSHOT.json]
python -m config_pipeline recover MANIFEST --db STATE.db [--snapshot SNAPSHOT.json]
python -m config_pipeline evolve OLD.json NEW.json --db STATE.db
python -m config_pipeline replay MANIFEST --db STATE.db
python -m config_pipeline diff OLD.json NEW.json
python -m config_pipeline compare LEFT.json RIGHT.json
python -m config_pipeline status --db STATE.db
python -m config_pipeline ui --db STATE.db
```

`--snapshot` 省略时默认写为 `STATE.db.snapshot.json`。`ui` 是只读 Tkinter 查看器；Tk 不可用时 CLI 核心功能不受影响。

`run` 可续跑；`recover` 会先标记未知分片，再重放当前已提交分片并继续剩余工作；`evolve` 只重算失效集合并整体替换当前视图。`--verify-replay` 可在 `run`/`evolve` 时额外重放校验。

测试专用强制终止注入：

```text
python -m config_pipeline run MANIFEST --db STATE.db --crash-after-completed 1
```

该参数在提交 N 个分片后，让下一分片保持运行中并直接 `exit(137)`，不输出退出报告。它用于验证恢复路径，不属于正常业务参数。

## 业务码与退出码

| 业务码 | 含义 | CLI 退出码 |
| --- | --- | --- |
| `E_MANIFEST` | 清单/修订/任务/端口等静态校验失败，含首个字段路径 | 2 |
| `E_PORT` | 声明回环端口无法绑定或已被占用 | 3 |
| `E_HANDSHAKE` | 真实套接字握手超时/子进程提前异常退出 | 4 |
| `E_REPLY` | JSONL 回执字段缺失、类型错误、集合摘要不一致或分发失败 | 5 |
| `E_DIVERGE` | 编译计划在稳定分片/字段处不一致 | 6 |
| `E_REPLAY` | 账本/快照完整性或恢复重放摘要不一致，含首个分片与字段 | 7 |

失败时 stdout 输出单个 JSON 对象：`{"ok":false,"code":"E_...","field":...}`。成功输出相应计划、状态或执行报告。

## 核心不变量

- **稳定计划**：集群按 ID、任务按 ID、键集合与目标集合排序；分片 ID 与计划序列化不依赖 JSON 对象或哈希遍历顺序。同一清单仅编辑字典/集合顺序会得到逐字节相同计划。
- **三重调度**：分片在提交给真实 `ThreadPoolExecutor` 前，先在主调度线程原子预占全局资源、集群运行数和任务运行数；工作线程结束后才释放。因此线程池排队也不能绕过预算。排队、开始、结束使用单调时钟毫秒。
- **选择性失效**：从修订链逐版本比较，沿 `配置键 -> 派生键 -> 任务引用 -> 目标集群 -> 分片` 传播，并附带可读依据。未失效分片保持原编译摘要、聚合摘要与来源修订；失效分片使用新修订重编译。
- **整体替换**：每次运行创建新世代，旧快照、视图、预算统计和失效轨迹全部设为 `is_stale=1`。当前视图只从唯一 current snapshot 读取，不做增量拼接。替换差异按稳定分片 ID 列出新增、删除和变化字段。
- **原子账本**：一个分片完成时，同一 SQLite 事务写入账本行、结局、资源单位、聚合摘要、修订版本、编译摘要和单调递增进度游标，同时更新账本内容摘要与 `inflight` 状态。强制终止只能看到完整前缀。
- **未知即重跑**：运行前先提交 `inflight=running`。重启时所有仍为 running 的尝试改为 unknown，不允许据此生成成功结论，并且对应分片必须重跑。
- **快照可验证**：快照保存集群集合、修订链、预算参数的内容版本摘要，以及分片轨迹、聚合摘要和账本游标。格式版本、版本摘要、内容摘要、账本游标或快照文件任一不一致时，整体重建当前视图。
- **重放校验**：恢复时按账本中的来源修订重新调用真实模拟服务，比较聚合摘要。差异按稳定分片顺序定位首个分片及嵌套字段，并以 `E_REPLAY` 说明依据。

更详细的表结构、状态机和模块边界见 [docs/DESIGN.md](docs/DESIGN.md)。实际执行结果见 [docs/VERIFICATION.md](docs/VERIFICATION.md)。

## 测试

```text
python -m unittest discover -v
```

当前工作区实际运行：32 项测试全部通过。覆盖清单首个字段错误、计划字节稳定性与 `E_DIVERGE`、三重预算、端口冲突 `E_PORT`、缺失回执 `E_REPLY`、派生失效传播、快照整体替换、真实子进程强制退出恢复和 `E_REPLAY`。

## 目录

```text
config_pipeline/       核心包（校验、编译、调度、模拟服务、SQLite、会话、CLI、Tk）
tests/                 标准库 unittest 测试
scripts/generate_samples.py  本地样本与 NDJSON 生成器
samples/               已生成的示例清单与请求
docs/                  设计与验证记录
```
