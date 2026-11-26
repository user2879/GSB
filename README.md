# Shard Diag：本地分片下发、租约、回滚与重放账本

一个完全本地、无外部服务的 Python/Linux(WSL) 组件。每个节点是独立子进程，在 `127.0.0.1` 真实 TCP 端口上说 JSON Lines；主进程通过线程池并行调度，维护预算/租约/并发三重约束、单调时钟租约、配额回滚失效传播、定点水位误差和 SQLite 原子账本。

原始需求逐条保留在 `docs/REQUIREMENTS.md`，实际验证记录见 `docs/VALIDATION.md`。

## 快速开始

```bash
./scripts/bootstrap.sh
./scripts/demo.sh
```

等价的手动命令：

```bash
python3 --version
python3 -m venv .venv
. .venv/bin/activate
python -m pip install numpy==1.26.4

python -m shard_diag.cli doctor
python -m shard_diag.cli validate --spec examples/cluster.json --events examples/events.json
python -m shard_diag.cli ports    --spec examples/cluster.json
python -m shard_diag.cli run      --spec examples/cluster.json --events examples/events.json \
  --ledger .runtime/demo.sqlite --output .runtime/demo-run.json
python -m shard_diag.cli verify   --spec examples/cluster.json --events examples/events.json \
  --ledger .runtime/demo.sqlite
python -m shard_diag.cli recompute --spec examples/cluster.json \
  --old-events examples/events.json --new-events examples/events_priority_edit.json \
  --ledger .runtime/demo.sqlite
python -m unittest discover -s tests
```

无图形显示环境不需要启动 Tk 窗口；`doctor` 只验证 `tkinter` 可导入，所有验收命令均为 CLI。

## 主要命令

- `doctor`：检查 `socket`、`threading`、`sqlite3`、`numpy==1.26.4`、`tkinter`。
- `validate`：强校验节点、分片、DAG、预算和事件流，输出声明摘要。
- `ports`：运行前确认所有端口是 `127.0.0.1` 且未被占用。
- `samples`：预生成所有“节点 × 分片”的确定性样本和小端原始字节文件。
- `run`：启动真实节点子进程、线程池调度、TCP/JSONL 聚合、租约、回滚和 SQLite 账本。
- `verify`：只读账本与原始声明，独立重推样本摘要、容量、租约版本、回滚传播、游标和内容摘要。
- `recompute`：比较新旧事件流，只对影响集合调用 NumPy，未影响分片复用账本摘要。
- `recover`：把有授予记录但没有原子结局的分片标记为 `unknown` 并要求重跑。
- `snapshot`：用 SQLite backup API 导出会话快照。
- `water`：输出单分片量化误差、节点水位比较界和全局预算误差归因。

## 错误码

| 码 | 触发点 |
| --- | --- |
| `E_SPEC` | JSON schema、标识/端口/整数约束、引用、DAG 或事件流不合法；返回首个字段和对象标识。 |
| `E_PORT` | 回环端口被占用、无法绑定或节点未在声明端口就绪。 |
| `E_TIMEOUT` | 真实 TCP 连接、握手或 JSONL 读取超过套接字超时。 |
| `E_FIELD` | 握手/回执缺少必需字段或 JSONL 非法。 |
| `E_LEASE` | 租约版本不符、已过期、非活跃或节点侧乐观代数校验失败。 |
| `E_VERIFY` | 独立重放发现摘要、租约、容量、回滚、游标或内容摘要不一致；定位首个分片和字段。 |

## 确定性聚合规则

1. 由节点整数种子和分片 ID 派生混合种子：`sha256("shard-diag-sample-v1\n<seed>\n<shard_id>")[:8]`。
2. 使用 `numpy.random.RandomState(mixed_seed)` 生成 16 个分组的 `group_index:int64` 和 `measure:int64`。
3. 测量值乘会话定点缩放因子；乘法为精确整数乘法，超出 signed int64 时饱和到 `±(2^63-1)`。
4. NumPy `np.add.at` 分组累加计数与定点和；批次只产生可相加的 `counts[16]`、`sums_q[16]`。
5. 最终摘要规范化为紧凑 JSON 后计算 SHA-256，绑定 `caliber_version`、scale、节点、分片、样本行数、参与行数和配额版本。

因此不同批次大小、下发顺序或工作线程并发度只改变部分聚合到达顺序，不改变可加总后的摘要或样本字节。

## 定点水位与误差

- 原始资源单位：`q = units * scale`，精确乘法，溢出饱和 int64。
- 比率水位：`q = saturate(floor(numerator * scale / denominator))`；非负数向零/向下截断。
- 每个分片的归一化预算份额有 `< 1 ulp` 的截断误差。
- `n` 个参与方的份额求和误差上界为 `< n ulp`，整数进位最多 `n-1 ulp`；报告拆分每个参与方余数和聚合进位。
- 两个定点水位直接比较的保守误差界为 `2 ulp`；预算剩余份额与候选需求份额同样使用 `2 ulp`。
- 落在误差界内时，不宣称水位优势，按“优先级降序 → Kahn 稳定拓扑序 → 分片 ID 字典序”决胜，并设置 `error_affected=true`。

## 账本与恢复

账本路径：`.runtime/demo.sqlite`。核心表：

- `shard_outcomes`：分片结局、聚合 JSON/摘要、租约归属、资源前后占用、水位、等待/执行单调时钟毫秒。
- `lease_events`：授予、续约、回收、取消、回滚撤销和失败裁决。
- `watermarks`：全局/节点水位和完整误差报告。
- `event_journal`：按到达序号保存事件哈希、事件体和放行/排队/回滚决策。
- `progress`：整数游标和状态。
- `recovery_records`：崩溃恢复出的 `unknown` 分片。

`meta.format_version` 是整数账本格式版本，`meta.spec_digest` 绑定当前声明，`meta.content_digest` 是所有核心表完整规范化行的根摘要。版本或声明摘要不匹配时，`Ledger.open_session` 会整体替换数据库，不做增量合并。

## 进程与端口清理

`NodeManager` 用 `start_new_session=True` 启动节点；强杀时对整个进程组发 `SIGKILL` 并 `wait()`，随后重新 `bind()` 声明端口确认没有孤儿监听。测试 `tests/test_protocol.py` 覆盖了长请求进行中强杀、端口释放和重启。

当前沙箱没有免密 sudo，且基础系统缺 `libtk8.6.so`。已用 `scripts/install-local-tk.sh` 的无 root 替代方案把 `libtcl8.6`、`libtk8.6`、`libxft2`、`libxss1` 解包到 `.sysdeps/root`；`doctor` 会用 `ctypes.RTLD_GLOBAL` 预加载。标准 WSL 推荐直接执行 `sudo apt-get install -y python3-tk`。

## 代码地图

- `shard_diag/spec.py`：JSON schema、强校验、DAG/稳定拓扑序。
- `shard_diag/determinism.py`：确定性样本、NumPy 分组聚合和摘要。
- `shard_diag/watermark.py`：定点量化、误差界和稳定决胜键。
- `shard_diag/protocol.py`：真实 TCP JSONL 握手/读写和错误码。
- `shard_diag/node_worker.py`：节点子进程入口与租约代数校验。
- `shard_diag/node_manager.py`：子进程启动、就绪探测、强杀和端口释放。
- `shard_diag/lease.py`：租约状态机、容量裁决和回滚 CAS。
- `shard_diag/engine.py`：线程池调度、等待计时、事件处理和原子结局。
- `shard_diag/ledger.py`：SQLite schema、事务、根摘要和整库替换。
- `shard_diag/impact.py`：编辑影响面、依赖传播和稳定失效序。
- `shard_diag/incremental.py`：只重算受影响摘要/水位。
- `shard_diag/verifier.py`：独立只读重放与 `E_VERIFY` 定位。
- `tests/`：27 个 unittest 回归，覆盖异常码、真实子进程、容量、过期、回滚、恢复和篡改定位。
