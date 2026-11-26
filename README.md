# GSB 本地多节点分片下发与恢复实验室

纯本地、无 Docker、无外部账号的 Python 3.10+ 项目。节点是真实独立子进程，在 `127.0.0.1` 上使用逐行 JSON（JSON Lines）通信；父进程用线程池调度，并实现租约、三重配额、定点水位裁决、失效增量重算、SQLite 原子账本、`SIGKILL` 后恢复和独立重放校验。

## 快速开始

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 如果 Debian/Ubuntu/WSL 提示 ensurepip 不可用：
# sudo apt install python3.11-venv python3-tk
# 或使用仓库已验证的 virtualenv：virtualenv .venv && .venv/bin/pip install -r requirements.txt

python -m gsb validate samples/local_spec.json
python -m gsb ports samples/local_spec.json
python -m gsb run samples/local_spec.json \
  --db artifacts/demo/ledger.db \
  --snapshot artifacts/demo/snapshot.json \
  --log-dir artifacts/demo/node-logs
python -m gsb verify --spec samples/local_spec.json --db artifacts/demo/ledger.db
```

一键验收：

```bash
bash scripts/acceptance.sh
# 或指定输出目录
bash scripts/acceptance.sh artifacts/acceptance-latest
```

最近一次实际结果：25 个测试全部通过；样例 6 分片真实 TCP 运行、独立校验、容量版本演进和增量/全量比对均成功。完整记录见 `artifacts/validation-report.txt`。

## 常用命令

- `python -m gsb validate samples/local_spec.json`：强校验 JSON 声明，失败输出 `E_SPEC`、首个 `field` 与 `position`。
- `python -m gsb ports samples/local_spec.json`：监听前检查回环端口冲突，占用输出 `E_PORT`。
- `python -m gsb run <spec> --db <sqlite> [--snapshot <json>] [--log-dir <dir>]`：启动节点子进程并在线程池中完成调度。
- `python -m gsb verify --spec <spec> --db <sqlite>`：以只读 SQLite 连接和原始声明重推全部结局，首个差异输出 `E_VERIFY`。
- `python -m gsb recover --spec <spec> --db <sqlite>`：回收上次进程遗留的 active 租约并列出未知分片。
- `python -m gsb evolve --old <old.json> --new <new.json> --workdir <dir>`：旧账本增量演进，并与空白全量重建逐字段比较，差异输出 `E_DIVERGE`。
- `python -m gsb fixedpoint-report`：输出定点误差三段归因和预算边界在不同缩放因子下的翻转。
- `python -m gsb gui --snapshot <json>`：可选 Tkinter 快照查看器；当前最小 WSL 镜像缺少 `libtk8.6.so` 时会返回 `E_TK` 和安装提示。

也可以安装控制台命令：`pip install -e .` 后使用 `gsb ...`。

## 声明样例

`samples/local_spec.json` 包含 3 个节点、6 个 DAG 分片、全局预算 40。初始两个分片占用 29 单位，第三个分片需要 12 单位，因此会真实经历 `global_budget` 排队和等待计时。`samples/evolved_capacity_spec.json` 将 `n3` 的容量版本从 2 提升到 3，用于验证失效传播。

支持的关键字段：

- 节点：`node_id`、`host`（可省略，必须为 `127.0.0.1`）、`port`、`concurrency`/`concurrency_quota`、`lease_limit`/`max_leases`、`seed`、`capacity_version`、可选 `fixed_scale`、`processing_delay_ms`。
- 分片：`shard_id`、`dependencies`/`depends_on`、`rows`、`resource_units`/`units`、`priority`、可选 `node_id`/`node_ref`。
- 预算：整数 `budget` 或 `{"budget": {"resource_units": N}}`；调度边界在 `scheduling.round_boundary`、`round_boundary_ms` 中声明。

## 错误码

- `E_SPEC`：声明、引用、唯一性、DAG、整数范围或缩放因子不合法。
- `E_PORT`：回环端口无法绑定或已被占用。
- `E_TIMEOUT`：TCP 建连、握手或 JSON-line 读取超时。
- `E_LEASE`：租约版本不符、状态不可续约或已过期。
- `E_DIVERGE`：增量路径与空白全量重建的首个字段差异。
- `E_VERIFY`：只读账本重放发现首个不一致。

## 代码地图

- `gsb/spec.py`：JSON 加载与强校验。
- `gsb/graph.py`：稳定拓扑序、节点引用和失效传播。
- `gsb/fixedpoint.py`：定点量化、饱和规则、误差界和边界裁决。
- `gsb/node.py`：节点子进程入口、NumPy 聚合和 JSON-lines 服务。
- `gsb/cluster.py`：子进程启动、端口/握手超时和 TCP 调用。
- `gsb/lease.py`：租约版本、单调时钟过期和一次性容量释放。
- `gsb/scheduler.py`：`ThreadPoolExecutor`、预算/租约/并发三重约束。
- `gsb/ledger.py`：SQLite 原子事务、游标、占用事件和恢复。
- `gsb/snapshot.py`：带格式版本/摘要的快照与整体替换视图。
- `gsb/reconcile.py`：失效子集增量重算和全量一致性比较。
- `gsb/verify.py`：只读独立重放校验。
- `tests/`：25 个单元与集成测试，包含真实超时、真实 TCP 和 `SIGKILL` 恢复。

数学口径、误差拆分和恢复协议见 `docs/TECHNICAL.md`。
