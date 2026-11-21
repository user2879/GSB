# 本地分布式副本修复组件

这是一个不依赖真实集群、外部账号或容器环境的 Python 组件，用于演示副本清单 v1/v2/v3 的互操作、版本升级、回环 TCP 摘要/数据交换、确定性真值判定、预算约束修复、SQLite 账本换代和逐字段重放。

组件只监听 `127.0.0.1`，节点数据、清单、检查点和账本全部保存在本地。样本由固定规则生成，不包含时间戳或随机内容。

## 快速开始

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m replica_repair env
.\.venv\Scripts\python -m replica_repair demo --root .
```

Linux/macOS（Python 3.10+）：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m replica_repair env
python -m replica_repair demo --root .
```

`demo` 会在 `data/groups`、`nodes`、`ledger` 下重建确定性样本并执行完整链路。它只清理自己已知的三组样本、对应节点目录和账本，不递归删除任务根目录。

核心链路不依赖 `tkinter`。`python -m replica_repair view --ledger <db>` 是可选查看器；Python 安装缺少 Tk 时会返回 `E_ENV`，不影响格式、升级、修复、账本和重放。

## 目录结构

```text
data/groups/interop-v1.json        # v1 旧清单
data/groups/interop-v2.json        # v2 旧清单
data/groups/interop-v3.json        # v3 当前清单
nodes/<node_id>/blocks/<bid>.bin   # 节点持久化块文件
ledger/<group_id>.sqlite3          # 每组 SQLite 账本
data/demo_report.json              # demo 的确定性摘要
```

节点目录由 `<root>/nodes/<node_id>` 推导；块文件由清单中的 `node_id`、`block_id` 定位。块数据先写同目录临时文件，`fsync` 后通过 `os.replace` 原子替换。

## 清单版本矩阵

| 字段 | v1 | v2 | v3 |
|---|---:|---:|---:|
| 顶层 `format_version`/`group_id`/`manifest_version` | 是 | 是 | 是 |
| 节点 `node_id` | 是 | 是 | 是 |
| 节点 `priority` | 否 | 是 | 是 |
| 块 `block_id`/`size_bytes` | 是 | 是 | 是 |
| 块 `digest`（`sha256:<64 hex>`） | 否 | 是 | 是 |
| 块 `repair_weight` | 否 | 是 | 是 |
| 块 `checksum_derivation` | 否 | 否 | 是，可为 `null` |
| 块 `repair_source` | 否 | 否 | 是，`origin` 或来源节点标识 |

v2 在版本演进中同时承载摘要、节点优先级和块修复权重。v3 的派生关系结构为：

```json
{
  "algorithm": "sha256-concat-v1",
  "source_block_ids": ["blk-0001", "blk-0002"],
  "digest": "sha256:..."
}
```

来源块必须在同一节点存在、按标识升序且无环。关系摘要计算方式为：

```text
sha256(canonical_json({
  "algorithm": "sha256-concat-v1",
  "sources": [{"block_id": ..., "digest": ...}, ...]
}))
```

v1 升级时从持久化块文件计算摘要，节点优先级与块修复权重均取 `1`；v2 升级时保留摘要和权重，补充 `checksum_derivation=null`、`repair_source="origin"`。所有成功升级最终都原子写成 v3，`manifest_version` 至少递增 1。

### `E_FORMAT`

未知版本、必需字段缺失、未知字段、非法摘要、非正整数大小/优先级/权重、重复标识、派生关系缺块/自引用/环/摘要不符均整体拒绝。错误对象包含 `path`，例如：

```json
{
  "ok": false,
  "error": {
    "code": "E_FORMAT",
    "message": "digest must be sha256:<64 lowercase hex>",
    "path": "$.nodes[0].blocks[0].digest"
  }
}
```

## 升级中断与续做

打开旧清单时执行“记录级升级”：

1. 原文件原子复制为 `<manifest>.v<old>.readonly.bak`，并移除写权限。
2. 创建 `<manifest>.upgrade.checkpoint.json`，按节点标识稳定排序。
3. 每升级一个节点，先将已完成节点的当前版本记录写入检查点，使用临时文件 + `os.replace` 原子提交，再处理下一个节点。
4. 全部记录提交后，完整 v3 清单原子替换目标文件；检查点标记 `complete`。
5. 重启时只接受检查点中“已提交前缀”。已提交记录直接从检查点恢复，不再次读取或升级；未提交记录才继续处理。

可确定性模拟强制终止：

```bash
python -m replica_repair upgrade --root . --crash-after 1 data/groups/interop-v1.json
# 进程以 86 退出；再次执行同一命令即从检查点续做
```

## 回环节点协议

节点服务只绑定 `127.0.0.1`。显式端口必须不小于 1024；端口 `0` 由操作系统选择空闲高位端口。绑定失败返回 `E_NETWORK`。

每行一个 JSON 请求，响应也是一行 JSON：

- `ping` / `health`
- `get_digest` + `block_id`
- `get_block` + `block_id`（base64 数据）
- `put_block` + `block_id`、`size_bytes`、`digest`、`data_b64`
- `reload_manifest`

节点每次摘要查询都会读取磁盘文件、检查清单字节数并重新计算 SHA-256。协调端随后拉取数据，再次计算摘要；返回声明摘要、实际内容和清单摘要不一致时以 `E_INTEGRITY` 拒绝。文件截断/超长以 `E_BLOCK` 拒绝。节点被终止后重启只依赖清单和块文件恢复。

## 真值判定

每个块先按节点标识排序，所有副本都通过拉取字节后的摘要校验。候选摘要的稳定排序键为：

1. 摘要频次降序；
2. 支持节点优先级权重和降序；
3. 支持节点标识向量字典序升序；
4. 摘要字符串升序。

排序第一的摘要成为真值。输出包含真值来源节点、全部副本、每个失配副本的实际偏离摘要和完整候选排名。所有列表均显式排序，JSON 使用 `sort_keys=True`，因此相同输入重复运行结果逐字节一致。

## 预算内修复

- `max_parallel`：全局同时执行任务上限。
- `per_node_kb`：每个目标节点在**单个调度轮次**内可用的修复 KB 配额。
- `cost_per_kb`：成本单位换算，默认 1。

任务按块修复权重降序、块标识、目标节点、源节点稳定排序。每轮按顺序装入任务；并发已满或目标节点配额不足时，任务进入下一轮并写入 `queue_events`。单任务大于 `per_node_kb` 时无法推进，直接以 `E_BUDGET` 拒绝，杜绝绕过配额。

修复量按块字节数向上取整为整数 KB。任务输出源节点、目标节点、块标识、`repair_kb` 和 `cost_units`。修复后目标清单更新摘要和 `repair_source`，并重算受影响的派生关系摘要。

## 账本与陈旧防护

SQLite 账本包含整数格式版本、当前磁盘内容摘要、修复前/后快照、副本分布、真值、失配集合、修复任务、进度游标和陈旧影响。

- `run` 先写修复前快照，再按任务提交进度；最后写修复后快照，并让它成为唯一当前视图。
- 重新打开时比较账本格式版本、当前清单和所有持久化块字节的状态摘要。
- 版本或内容不一致时，旧快照全部标记 `is_stale=1`，未完成任务标记 `unknown`，当前视图置空；下次完整运行写入新的前/后快照，不把旧行合并进当前视图。
- 优先级、清单版本或派生关系变化后，`ManifestRepository` 保证 `manifest_version` 单调递增，并稳定输出受影响块/节点/原因。
- `report` 若账本绑定了本地清单路径，会先自动执行磁盘状态复检；没有绑定的只读账本无法推断源目录。

## 重放与异常

重放会重新计算当前块摘要、当前真值、修复前真值及预算内计划，并与账本逐项比较。首个差异通过 `E_REPLAY.details.first_mismatch` 给出：

```json
{
  "block_id": "blk-0002",
  "node_id": "v3-a",
  "field": "digest",
  "ledger_value": "sha256:...",
  "recomputed_value": "sha256:...",
  "basis": "sha256 recomputation from persisted block bytes and current manifest"
}
```

异常注入：

```bash
python -m replica_repair inject --root . truncate v3-a:blk-0001
python -m replica_repair run --root . --manifest data/groups/interop-v3.json \
  --ledger ledger/interop-v3.sqlite3          # E_BLOCK

python -m replica_repair inject --root . corrupt v3-b:blk-0001
python -m replica_repair run --root . --manifest data/groups/interop-v3.json \
  --ledger ledger/interop-v3.sqlite3          # E_INTEGRITY

python -m replica_repair run --root . --manifest data/groups/interop-v3.json \
  --ledger ledger/interop-v3.sqlite3 --max-parallel 0   # E_BUDGET
```

也可直接给 `run` 传 `--inject-truncate v3-a:blk-0001` 或 `--inject-corrupt v3-b:blk-0001`。

## 常用命令

```bash
python -m replica_repair demo --root .
python -m replica_repair validate data/groups/interop-v3.json
python -m replica_repair upgrade --root . data/groups/interop-v1.json
python -m replica_repair serve-node --root . --manifest data/groups/interop-v3.json --node v3-a --port 0
python -m replica_repair run --root . --manifest data/groups/interop-v3.json \
  --ledger ledger/interop-v3.sqlite3 --max-parallel 3 --per-node-kb 3
python -m replica_repair report --ledger ledger/interop-v3.sqlite3
python -m replica_repair replay --root . --manifest data/groups/interop-v3.json \
  --ledger ledger/interop-v3.sqlite3
python -m replica_repair evolve --manifest data/groups/interop-v3.json --set-priority v3-c=6
```

退出码：`2=E_FORMAT`、`3=E_BLOCK`、`4=E_INTEGRITY`、`5=E_BUDGET`、`6=E_REPLAY`、`7=E_STATE`、`8=E_NETWORK`、`9=E_ENV`。

## 测试

```bash
python -m unittest discover -s tests -v
```

详细设计见 `docs/TECHNICAL.md`，本次环境实际验证结果见 `docs/VERIFICATION.md`。

## 当前环境说明

- 本工作区使用 Python 3.11、SQLite 3.37、NumPy 1.26。
- 当前 Linux/WSL 容器的 Python 未安装 Tk；Tk 查看器已做成可选入口，Windows 官方 Python 通常自带 Tk，可先运行 `python -m tkinter` 确认。
- 所有网络交互仅限 `127.0.0.1`，未访问外部目标。
