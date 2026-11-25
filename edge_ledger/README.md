# Edge Ledger：本地多节点 CDN 分片校验账本

首版在空工作区中建立，全部组件均运行在 WSL Ubuntu 的 `127.0.0.1` 回环上：

- 节点是真实 `subprocess.Popen` 子进程，使用 JSON Lines over TCP。
- 调度使用真实 `ThreadPoolExecutor`、真实套接字超时和 SQLite 事务。
- 内容由固定整数种子经 NumPy `RandomState` 确定性生成。
- v1 定宽文本、v2 JSON、v3 JSON 统一归一到可复核账本。
- 到期回收与配额回收通过 SQLite CAS + 唯一部分索引给出唯一归属。
- `SIGKILL` 节点后只依据已提交记录恢复，未完成分片标记 `unknown` 并先回收容量。

## 环境

已验证环境：

- `/usr/bin/python3.10`：Python 3.10.12。
- 项目虚拟环境：Python 3.11.0rc1。
- numpy 2.4.6、pytest 9.2.0、sqlite3 3.37.2。
- WSLg `DISPLAY=:0` 可启动真实 Tk 窗口。

本机缺少 `ensurepip` 和系统 Tk 动态库，因此使用 `virtualenv` 引导：

```bash
python3 -m virtualenv .venv
.venv/bin/python -m pip install numpy pytest
.venv/bin/python -m pip install -e .
```

若系统 Tk 缺少 `libtk8.6.so`，可把 Ubuntu jammy 的 `libtk8.6`、`libtcl8.6`、
`libxft2`、`libxss1` 等 deb 解压到 `.sysroot/`。`scripts/edge-ui.sh` 和
`scripts/run-acceptance.sh` 会自动设置 `LD_LIBRARY_PATH`、`TCL_LIBRARY`、
`TK_LIBRARY`，并在需要时加入基础解释器中的 `tkinter` 包目录。没有使用 pip 安装 Tk。

## 快速运行

```bash
cd edge_ledger

# 生成确定性声明、事件流和三版回执
.venv/bin/edge-ledger-samples --out samples --fixed-ports

# 运行全部单元/集成测试（包含真实子进程验收流）
.venv/bin/pytest -q

# 一键端到端验收；加 --tk-smoke 会启动并自动关闭两次 Tk 窗口
./scripts/run-acceptance.sh

# 独立只读重放核对
.venv/bin/edge-ledger-verify \
  --spec-dir artifacts/latest-validation/spec-v2 \
  --ledger artifacts/latest-validation/edge.db

# Tkinter 诊断界面
./scripts/edge-ui.sh \
  --spec-dir <run-dir>/spec-v2 \
  --ledger <run-dir>/ledger-unknown-state.db

# 无窗口时输出稳定文本界面快照
.venv/bin/edge-ledger-ui \
  --spec-dir <run-dir>/spec-v2 \
  --ledger <run-dir>/ledger-unknown-state.db \
  --text-snapshot /tmp/ui.txt
```

最新一次验收的稳定摘录在 `artifacts/latest-validation/`：

- `acceptance-report.json`：完整机器可读结果。
- `replay-report.json`：独立重放报告。
- `environment.txt`：Python/依赖/回环端口证据。
- `ui-unknown-snapshot.txt`：含 `UNKNOWN/RERUN` 标记的界面快照。
- `ui-final-snapshot.txt`：重跑恢复后的界面快照。
- `ui-unknown-tk.txt`、`ui-final-tk.txt`：真实 Tk 冒烟退出码，均为 `0`。

## 目录结构

```text
src/edge_ledger/
  spec.py          # E_SPEC 强校验、SpecBundle、只读原始快照
  jsonloc.py       # JSON 路径到源文件行号映射
  integrity.py     # NumPy 确定性字节、定点统计、摘要绑定
  formats.py       # v1/v2/v3 编解码、补齐依据、E_FORMAT
  ledger.py        # SQLite schema、事务、容量/预算、CAS、备份替换
  leases.py        # 租约、队列、真实到期、配额演进、唯一裁决
  protocol.py      # 回环 JSON-line 客户端和套接字超时
  node_worker.py   # 可 SIGKILL 的节点子进程入口
  runtime.py       # Popen 监管、强杀、重启
  scheduler.py     # ThreadPoolExecutor 调度和配额不变量
  replay.py        # 只读事件折叠与 E_REPLAY 核对
  dashboard.py     # 稳定排序的 UI 数据模型/文本快照
  ui.py            # Tkinter/ttk 诊断界面
  acceptance.py    # 本地一键验收工作流
tests/             # 16 个单元与集成测试
samples/           # 已生成的确定性样本
```

## 关键规则

### 规格校验

`nodes.json`、`content_manifest.json`、`lease_policy.json`、`event_stream.jsonl`
必须满足：

- `node_id`、`content_key` 唯一。
- `host` 必须是字面量 `127.0.0.1`，`host:port` 全局唯一。
- `capacity_slots`、`max_node_leases`、`integer_scale` 拒绝布尔值和负数。
- `integer_scale` 只允许 `(1, 10, 100, 1000)`。
- `shard_count` 必须整除 `total_bytes`。
- 事件 `seq` 严格递增。

错误对象包含 `code=E_SPEC`、首个非法 `field` 和源文件 `line`。
`raw_snapshot/` 内文件权限为 `0444`。

### 三版回执

- v1：32 字符 `shard_id` + 40 字符 `content_key` + 64 字符摘要，整行 136 字符。
- v2：JSON 增加 `shard_seq`、`sample_bytes`、`node_id`。
- v3：JSON 增加 `lease_version`、`quota_version`、`invalid_shard_ref`、
  `content_bytes_digest`。

归一结果记录 `source_format`、排序后的 `filled_fields` 和逐字段 `fill_basis`。
所有摘要使用 `sort_keys=True` 的规范 JSON，不依赖字典或哈希遍历顺序。

### 确定性校验

分片种子推导：

```text
seed32 = first_32_hex(sha256(
  "edge-content/v1|{node_seed}|{manifest_seed}|{content_key}|{shard_seq}"
))
bytes = numpy.random.RandomState(seed32).randint(0, 256, shard_size, dtype=uint8)
```

定点口径按 `sample_stride` 取样，使用位置权重
`w_i = (i mod 17) + 1`，计算
`q = floor(scale * weighted_sum / total_weight)` 和非负余数。
最终摘要绑定节点、内容键、分片序号、字节区间、定点统计和原始字节摘要。

### 配额与裁决

授予前同时检查：

- 全局活跃租约 `< global_max_active_leases`。
- 节点活跃租约 `< min(capacity_slots, max_node_leases)`。
- `bytes_verified + bytes_reserved + byte_count <= global_byte_budget`。

无法立即获得额度时进入条件变量队列，并写 `queue_wait_started` /
`queue_wait_ended` 的单调时钟毫秒。到期判断为
`monotonic_ms() >= expires_ms`；轮询线程只负责及时回收，不冒充到期条件。

回收使用条件更新：

```sql
UPDATE leases
SET state = ?
WHERE lease_id = ? AND state = 'active';
```

只有 `rowcount == 1` 的调用方成为唯一赢家。部分唯一索引还保证同一 quota/node/slot
和同一 shard 同时只有一个活跃租约。

### 恢复与快照

账本元数据保存整数 `ledger_format_version=1` 和 `content_digest`。重新打开时若
版本或摘要不一致，旧库被 `os.replace` 到 `backups/*.db` 并 `chmod 0444`，随后
建立全新账本，不做增量合并。

节点被 `SIGKILL` 后：

- 已提交的摘要保留。
- 已提交授予但没有已提交结果的租约由 `recover_active_leases()` 原子终结。
- 槽位和字节预留先释放。
- 对应分片写为 `status=unknown, unknown=1`，必须重跑。
- 不生成也不承诺强杀时的退出报告。

## 独立核对

`edge_ledger.replay` 以 SQLite `mode=ro` 打开账本，只读取账本和原始声明：

1. 按 `event_seq` 折叠授予、续约、完成、到期、配额回收和恢复。
2. 比较活跃租约集合、节点、槽位、版本和到期时间。
3. 逐事件比较节点活跃数与 `capacity_margins`。
4. 用 NumPy 重新推导每个分片摘要和内容字节摘要。
5. 比较 `adjudications`、事件和租约终态中的裁决依据。

首个不一致以 `E_REPLAY` 给出 `shard_id`、`field` 和 `basis`。

## 界面

Tkinter 界面有三个标签页：

- Lease watermarks：节点槽位、有效租约上限、活跃数/容量、可用槽位。
- Shard conclusions：当前 quota 下的校验结论和 `UNKNOWN/RERUN`。
- Receipt decisions：三版回执的采纳/否决、替换、当前版本和 stale 标记。

界面只用只读 SQLite URI 打开账本；Refresh 不写入或改变账本。
