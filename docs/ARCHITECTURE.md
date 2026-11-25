# Shardflow 技术说明

## 1. 目标与边界

Shardflow 在单机 Windows（同时兼容 POSIX）上编排多个独立 Python 子进程节点，
在 `127.0.0.1` 回环 TCP 上以 JSON 行协议通信，对由整数种子确定性生成的样本执行
NumPy int64 定点分组聚合。系统不依赖外部服务、容器或真实账号。

核心能力：

- 节点、分片依赖图、配置修订链、事件流的 JSON/NDJSON 强校验。
- 真实子进程 + 真实套接字超时的逐行协议。
- 可续约、可真实过期、带乐观版本号的分片租约。
- 全局预算、节点租约上限、节点并发配额三重约束下的线程池并行调度。
- 依赖就绪、单次调度、取消终态与重复取消 `E_DUP`。
- 配置修订后的稳定拓扑序失效传播与仅失效子集重算。
- SQLite 单事务账本、格式版本、内容摘要、整体替换与强制终止恢复。
- 只读独立重放校验与 Tkinter 诊断界面。

## 2. 模块结构

| 模块 | 职责 |
| --- | --- |
| `shardflow/spec.py` | 声明加载、NDJSON 事件、强校验、稳定拓扑排序 |
| `shardflow/samples.py` | 整数种子样本生成、样本摘要、int64 定点聚合 |
| `shardflow/protocol.py` | JSON 行协议、节点 TCP 服务端、节点客户端 |
| `shardflow/node.py` | `python -m shardflow.node` 子进程入口 |
| `shardflow/process.py` | Windows/POSIX 子进程生命周期与握手 |
| `shardflow/lease.py` | 租约值对象、确定性租约 ID、纯函数续约规则 |
| `shardflow/engine.py` | 调度、租约守护、事件处理、失效重算、恢复 |
| `shardflow/invalidation.py` | 稳定拓扑序失效传播与每步依据 |
| `shardflow/ledger.py` | SQLite 事务、格式版本、内容摘要、整体替换 |
| `shardflow/replay.py` | 仅读账本与声明的独立重放校验器 |
| `shardflow/viewmodel.py` | 无界面可测的稳定只读诊断视图模型 |
| `shardflow/diagnostics.py` | Tkinter 界面（延迟导入 Tk） |
| `shardflow/cli.py` | `validate/run/verify/inspect/diagnose` 命令 |

## 3. 数据与协议

### 3.1 声明文件

- `nodes`：`node_id`、字面量 `host=127.0.0.1`、全局唯一 `port`、正整数
  `concurrency`、`lease_limit`、`seed`、`capacity_version`、`fixed_scale`、
  非负 `max_renewals` 与 `compute_iters`。
- `shards`：唯一 `shard_id`、目标节点、`depends_on`、正整数 `resource_units`、
  `sample_count` 以及该分片读取的 `config_keys`。
- `revisions`：单调递增正整数 `version`、显式 `parent`（根为 `null`）、整数键值。
- 事件 NDJSON：`dispatch`、`cancel`、`quota_change`、`config_apply`。

首个违规对象与字段通过 `E_SPEC` 报告。取消事件必须指向此前已下发的分片；
运行时重复取消或对终态分片取消以 `E_DUP` 报告。

### 3.2 JSON 行消息

- 客户端到节点：`handshake`、`dispatch`、`renew`、`cancel`。
- 节点到客户端：`handshake_ack`、`receipt`、`renew_ack`、`renew_reject`、
  `cancel_ack`。
- 每条消息为一个 JSON 对象加 `\n`；套接字超时始终显式设置。
- 端口不可用/无法连接为 `E_PORT`；握手窗口内未完成握手为 `E_HANDSHAKE`；
  回执缺字段为 `E_FIELD`。

### 3.3 定点聚合

样本由 `numpy.random.default_rng(node.seed)` 生成，分组与值均为 int64。
聚合使用 `np.add.at` 进行无缓冲重复索引累加，乘以节点定点尺度与修订整数因子，
输出每组 count/sum/average、样本数、修订因子模、burn-in 总和与摘要 digest。
整个过程不使用浮点聚合值，相同输入在不同进程与运行间字节一致。

## 4. 调度与租约

- 调度器以 20ms 周期运行，每轮按稳定拓扑序扫描，每个分片每轮最多授予一次。
- 授予条件同时满足：
  - `在途资源单位之和 + 分片单位 <= global_budget`；
  - 节点活跃租约数 `< lease_limit`；
  - 节点活跃分片数 `< 当前并发配额`（含 `quota_change` 演进）；
  - 全部分依赖已完成且聚合摘要可用。
- 不满足条件时记录单调时钟 `wait_start_ms/wait_end_ms`，不睡眠等待。
- 实际计算投射到 `ThreadPoolExecutor`，由真实节点子进程并行执行。
- 租约包含 `lease_id/shard_id/node_id/granted_ms/expires_ms/renewals/
  lease_version/attempt/renewed_ms`。
- 续约先在引擎侧做纯函数乐观校验，再向节点验证版本；版本不符或达到上限返回
  `E_LEASE`，且不延长到期时刻。
- 回收线程以单调时钟比较 `now >= expires_ms`，没有人为延迟来模拟时间。
- 已取消分片为终态；其下游若依赖被取消分片，会稳定进入 `blocked` 终态并落账。

## 5. 失效传播与子集重算

`config_apply` 事件触发以下步骤：

1. 折叠修订父链得到新旧整数配置映射。
2. 直接失效：分片声明的 `config_keys` 发生变化、节点容量版本变化或分片参数变化。
3. 间接失效：沿 `depends_on` 的反向依赖关系传播。
4. 输出按依赖图稳定拓扑序排列，并记录每步 `reason/basis`。
5. 已取消分片不重新下发；仅失效且非取消的分片重置并重算。
6. 未失效分片的结局、摘要 digest 与配置版本保持不变，可逐项字节比较。

节点子进程在配置应用后重启，以加载新的有效整数配置映射；该重启也作为节点
强制终止恢复路径的真实演练。

## 6. SQLite 账本与恢复

每个终端结局在同一个 SQLite 事务中写入：

- 分片结局与尝试次数；
- 聚合摘要与 outcome digest；
- 资源单位、租约归属、容量版本、配置版本；
- 等待/授予/完成单调时钟；
- 进度游标。

另有仅追加的租约事件表、失效依据表、快照表和元数据表。元数据携带整数
`format_version` 与覆盖全部业务表的 `content_digest`。

打开账本时：

- 格式版本不支持或内容摘要不一致：严格模式抛 `E_LEDGER`；默认模式删除数据库及
  WAL/SHM 后整体重建，不做增量合并。
- 强制终止后只信任已提交事务；未提交内容不会出现在 SQLite 中。测试通过显式
  `unknown` 行模拟“事务未完成”的分片，恢复时将其标记为未知并要求重跑，
  不生成所谓强制终止时刻退出报告。
- 恢复完全基于已提交记录重建内存视图，未知分片重跑后再次执行只读重放。

## 7. 独立重放

`shardflow verify <spec> --events <events> --ledger <db>` 不启动节点、不写账本：

1. 重放 grant/renew/expire/complete/cancel/invalidated/quota_change 事件；
2. 逐授予点检查全局预算、节点租约上限与并发配额；
3. 独立重新生成确定性样本并计算定点摘要；
4. 比较摘要、outcome digest、容量版本、配置版本、失效集合与传播依据；
5. 检查最终无活动租约、资源占用归零。

首个不一致以 `E_REPLAY` 报告，并定位到分片 ID 与字段名。

## 8. Tkinter 诊断

`shardflow diagnose <spec> --events <events> --ledger <db>` 打开 Tk 界面。
Tkinter 在函数内延迟导入，因此无 Tk 原生库的无头环境仍可运行其他命令和全部
逻辑测试。界面所有数据来自只读 `viewmodel.build_view_model`，刷新只重新读取，
不执行任何写事务。标签页包括节点租约/资源、分片状态、失效依据、未知分片与
账本快照，所有列表按节点 ID 或拓扑序稳定渲染。

## 9. Windows 安装与运行

```powershell
python --version          # 需要 3.11 或 3.12
python -m pip install numpy==1.26.4
python examples/generate_demo.py
python -m shardflow.cli validate demo/demo.spec.json --events demo/demo.events.ndjson
python -m shardflow.cli run demo/demo.spec.json --events demo/demo.events.ndjson --ledger demo/demo.ledger.sqlite
python -m shardflow.cli verify demo/demo.spec.json --events demo/demo.events.ndjson --ledger demo/demo.ledger.sqlite
python -m shardflow.cli diagnose demo/demo.spec.json --events demo/demo.events.ndjson --ledger demo/demo.ledger.sqlite
```

可重复演示：

```powershell
python examples/demo_run.py all
```

## 10. 已验证结果（Linux 沙箱 / Python 3.11 / NumPy 1.26.4）

- `python -m unittest discover -s tests`：32 个测试全部通过。
- `python examples/demo_run.py all`：`full`、`crash`、`tamper` 三模式全部成功。
- 完整运行后独立 `verify` 返回 `status=ok`，最终 `resource_in_use=0`。
- 强制终止恢复将 `s2` 标记为 unknown，重启节点后重跑为 completed，unknown 清零。
- 篡改 content digest 后打开账本触发整体替换，替换后租约事件数为 0。
- 当前沙箱缺少 Tk 原生库（`libtk8.6.so`），无法实际弹窗；Windows 官方 Python
  安装包自带 Tkinter。界面逻辑通过无界面 view model 的稳定性测试覆盖。
