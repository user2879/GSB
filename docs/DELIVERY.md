# ShardMesh 交付说明

首版实现为 `src/` 布局的可编辑 Python 包 `shardmesh`。它会真实启动节点子进程、绑定 `127.0.0.1` 高位端口、通过 TCP JSON Lines 通信、在线程池中并行调度，并把结局原子写入 SQLite 账本。

## 快速命令

```bash
# 标准环境若 ensurepip 可用
python3 -m venv .venv
source .venv/bin/activate
pip install numpy
pip install -e .

# 生成确定性 spec/samples
shardmesh init --dir artifacts/demo

# 运行真实三节点、8 分片、10 事件演示
shardmesh run --dir artifacts/demo

# 28 个 unittest：协议、租约、配额、误差、失效、快照、重放、SIGKILL
python -m unittest discover -s tests

# 增量推进 vs 空白全量重建；动态申请空闲端口
shardmesh verify --dir artifacts/verification --dynamic-ports

# SIGKILL n1，再按 SQLite 账本恢复、重跑、重放
shardmesh chaos --dir artifacts/chaos

# 可选 Tk 当前视图
shardmesh gui
```

也可直接运行 `scripts/run_demo.sh`。

## 实际环境记录

- 当前解释器：Python 3.11.0rc1，路径位于预构建的 `/home/zsl/.feedback/python-base`。
- 系统的 `python3 -m venv` 因缺少 `ensurepip/python3.11-venv` 失败；当前账号免密 sudo 不可用，因此使用环境已提供的 `virtualenv -p python3 .venv` 建立隔离环境。这是等价的本地虚拟环境，不使用 Docker。
- `.venv` 中实际安装：`numpy 2.4.6`，包本身以 editable 模式安装。
- 直接在当前 `.venv` 执行 `shardmesh gui` 会返回结构化 `E_GUI`：virtualenv 默认不暴露预构建解释器站点目录中的 `tkinter`，且系统最初缺少 `libtk8.6.so`、`libtcl8.6.so`、`libXft.so.2`、`libXss.so.1`。
- 验收时另用 `apt-get download` 把上述 Tk/X11 库解包到 `/tmp/tkroot`，并附加 `LD_LIBRARY_PATH/TCL_LIBRARY/TK_LIBRARY/PYTHONPATH` 后，`tk.Tk()` 创建成功，`shardmesh gui` 无异常进入 mainloop；`timeout 2s` 的退出码为 `124`。标准 WSL Ubuntu 推荐直接安装系统包：

```bash
sudo apt update
sudo apt install -y python3-venv python3-tk
```

若使用自编译/额外 PPA 的 Python 3.11，还需安装与其匹配的 `python3.11-tk` 或 Tk 运行库。GUI 代码本身仅使用标准库 `tkinter/ttk`。

## 声明 JSON

生成的 `artifacts/demo/spec.json` 包含：

- `global_budget=7`、`scale=1000`、`max_renewals=1`、`lease_ttl_ms=2000`。
- 3 个节点：`n1/n2/n3`，固定演示端口 `43181/43182/43183`，分别声明并发配额、租约上限、随机种子和容量版本。
- 8 个分片：包含题目列出的 `id/deps/rows/units/priority`；演示文件还显式给出 `node_id` 以精确表达“分片 → 节点参数”的失效引用。`node_id` 和节点 `host` 也是可选字段：缺省 host 为 `127.0.0.1`，缺省 node 按分片清单下标稳定轮询分配。
- 事件：8 个按 `seq` 严格递增的 `arrive`，随后是 `set_node_quota(n2,1)` 和 `bump_capacity_version(n3,2)`。

支持的参数事件：

- `arrive`
- `set_budget`
- `set_node_quota`
- `set_node_lease_limit`
- `set_scale`
- `set_max_renewals`
- `set_lease_ttl`
- `bump_capacity_version`

`scale` 仅接受 `(1, 10, 100, 1000, 10000)`。所有整数字段均拒绝布尔值，避免 Python `bool` 被静默当作 `0/1`。

## 需求与代码映射

| 主题 | 主要文件 | 落地内容 |
| --- | --- | --- |
| JSON 强校验 | `src/shardmesh/spec.py`、`graph.py` | 全量解析后才返回对象；ID/端口唯一、loopback 字面量、非负整数、依赖存在、DAG 无环、scale 集合、事件 seq 严格递增；错误为 `E_SPEC` 并带 `category/index/seq/field`。 |
| 样本与聚合 | `samples.py`、`node_worker.py` | 固定种子派生 NumPy 样本；`np.bincount/np.add.at` 分组计数与求和；回执含 `node_id/shard_id/capacity_version/summary/elapsed_ms`。 |
| TCP/子进程 | `cluster.py`、`protocol.py`、`node_worker.py` | `subprocess.Popen([sys.executable, "-m", "shardmesh.node_worker", ...])`；真实 `127.0.0.1` TCP、socket timeout、JSON line；`E_PORT/E_TIMEOUT/E_FIELD/E_NODE`。 |
| 租约 | `leases.py`、`engine.py` | 七要素租约、单调时钟 ns 截止时间、后台 reaper、真实过期、容量释放；续约先请求节点做版本乐观校验，成功后才本地升版本/延长期限。 |
| 并行调度 | `capacity.py`、`engine.py` | `ThreadPoolExecutor` 真实工作线程；全局 units、节点活跃分片、节点活跃租约三重硬约束；依赖必须有当前 summary；记录 wait start/end/ms。 |
| 定点水位 | `fixedpoint.py`、`water.py` | `q=floor(num*scale/den)`；非负除法为向零/向下截断；乘法溢出饱和到 `2^63-1`；输出逐分片截断余数、节点水位、预算边界和全局聚合上界。 |
| 误差归因 | `water.py` | 三段：`single_value_quantization_q=1`、`dependency_propagation_q=depth`、`aggregation_accumulation_q=1+direct_dependency_count`；输出 `synthetic_bound_q`、主导因和显式阈值。 |
| 边界决胜 | `engine.py` | 优先级后，若两候选 projected water 差不超过合成界，或预算 slack 落在误差界内，标注 `error_affected`，唯一选择 `(stable_topology_index, shard_id)` 最小者。 |
| 容量版本防护 | `engine.py`、`view.py`、`snapshot.py` | 版本递增时撤销该节点所有旧 cv 租约并释放配额；旧记录置 `stale/current=false`；当前视图整体重建，快照临时文件 + `os.replace` 原子替换。 |
| 失效传播 | `graph.py`、`engine.py` | 节点参数直接失效引用该节点的分片，再沿反向依赖 BFS；输出稳定拓扑序集合与每步 basis；只更新失效行，未失效 `record_json` 逐字节不变。 |
| 分歧校验 | `verification.py`、`ledger.py` | 同 spec/samples 跑增量和空白全量；核心语义投影递归逐字段比较，不一致报 `E_DIVERGE`，details 带 `shard_id/node_id/field_path`。 |
| SQLite 账本 | `ledger.py` | outcome 同一事务写记录、租约释放状态、cursor、ledger event；含整数 `format_version=1` 与 canonical content digest。 |
| 快照恢复 | `snapshot.py` | 打开时校验版本和摘要；失败则把旧快照复制为只读 `*.readonly-backup-N`（chmod `0444`），再整体重建替换。 |
| SIGKILL 重放 | `chaos.py`、`replay.py` | `process.kill()` 发 SIGKILL；重启只信已提交记录；active lease 标记回收，缺失 outcome 标 `unknown` 且要求 rerun；重算 NumPy summary 并逐项比较，失配报 `E_REPLAY`。 |
| GUI | `gui.py` | 标准库 Tkinter Treeview/Text，按钮触发 whole-view refresh；当前环境缺 Tk 运行库时返回 `E_GUI`。 |

## 错误码

| Code | 触发点 |
| --- | --- |
| `E_SPEC` | JSON schema、ID/端口、依赖图、scale、事件序列不合法。 |
| `E_PORT` | 端口不可绑定、占用、连接被拒或节点启动绑定失败。 |
| `E_TIMEOUT` | TCP 连接/握手/读取超过真实 socket timeout。 |
| `E_FIELD` | JSON 响应缺字段、字段类型不符或回执 schema 不完整。 |
| `E_LEASE` | 租约不存在、版本不符或已过期；不延长到期时间。 |
| `E_RENEW` | 续约次数超过 `max_renewals`。 |
| `E_DIVERGE` | 增量与全量重建的核心字段不一致；未失效账本字节变化也使用该码。 |
| `E_REPLAY` | SQLite 摘要、重放聚合或恢复记录逐项比较失败。 |
| `E_GUI` | 当前 Python/Tk 运行库不可用。 |
| `E_NODE` | 节点进程死亡、socket 重置等未被更具体错误码覆盖的节点故障。 |

## 可复现验证结果

下列结果来自本次工作区，命令输出同时保存在 `artifacts/validation/`：

- `python -m unittest discover -s tests`：`Ran 28 tests ... OK`。
- `shardmesh run --dir artifacts/demo`：`committed=8`，ledger replay `checked_count=8, consistent=true`，snapshot digest 校验匹配。
- `shardmesh verify --dir artifacts/verification --dynamic-ports`：`consistent=true`，未失效字节保持 `s01/s03/s06`，变更集合为 `s02/s04/s05/s07/s08`，两次参数事件均生成传播依据。
- `shardmesh chaos --dir artifacts/chaos`：`sigkill_returncode=-9`，恢复出 1 条未提交 active lease，未知分片包含 `s01`；重跑后 replay `checked_count=8, consistent=true`；`exit_report_promised=false`。
- `shardmesh gui`：当前沙箱返回 `E_GUI`，原因是系统 Tk/X11 共享库缺失；这不影响 CLI、TCP、调度和账本验收。

## 时钟与逐字段比较口径

租约和等待使用 `time.monotonic_ns()`，过期由真实睡眠时间触发，不使用模拟时钟推进。绝对 monotonic 时间戳、本次调度观察到的瞬时 active count/water q、节点毫秒耗时会随运行变化；这些值仍写入单次账本并由 `content_digest` 保护，但不适合跨两个独立进程做字面相等比较。

因此校验分两层：

1. 同一 SQLite 账本的恢复/重放：canonical digest 和全部已存储字段必须自洽，聚合摘要由确定性样本重算并逐字段比较，失配为 `E_REPLAY`。
2. 增量运行与空白全量运行：比较可复现语义投影，包括 node、capacity version、summary、lease ID/version、固定口径、误差归因、拓扑版本和裁决结果；未失效 SQLite `record_json` 则要求逐字节不变。

这保留了真实 monotonic 过期和真实线程/网络调度，同时避免把不同进程的启动后时钟读数误判为业务分叉。
