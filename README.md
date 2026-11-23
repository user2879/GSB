# Cacheproof

面向 Windows 本地环境的缓存段容量证明与读视图收敛诊断组件。它不处理锁、租约或分布式仲裁，只处理三件事：

- 缓存段每次写入/删除后，都保留一份可以独立复核的容量证明。
- 读视图与段当前状态不一致时，整行原子替换并给出稳定差异。
- 进程被强杀后，只从已提交账本恢复；未完成条目标为未知，不伪造退出报告。

实现只依赖 Python 标准库：`sqlite3`、`json`、`hashlib`、`tkinter` 等。

## Windows 环境

在 PowerShell 或 CMD 中执行：

```bat
python --version
python -m venv .venv
.venv\Scripts\activate
python --version
python -c "import sqlite3, tkinter; print(sqlite3.sqlite_version, tkinter.TkVersion)"
```

要求 Python 3.11 或更高版本，无需 `pip install`。若第二个检查在导入 `tkinter` 时失败，请重新运行 Python 安装程序并勾选 `tcl/tk and IDLE` 组件。

所有运行产物都限制在指定工作区内：SQLite 账本、内容文件、会话快照、日志、临时文件均位于项目目录下的工作区目录。程序拒绝文件系统根目录、Windows 系统保留目录和 Windows 保留文件名。默认工作区是 `.cacheproof`，演示默认使用 `demo-workspace`。

## 快速开始

```bat
python -m cacheproof init --workspace .cacheproof --spec samples\cache_spec.json
python -m cacheproof verify --workspace .cacheproof
python -m cacheproof show --workspace .cacheproof
python -m cacheproof put --workspace .cacheproof --segment seg-alpha --key alpha/new --size 32 --seed alpha-new-v1
python -m cacheproof view --workspace .cacheproof --view-id view-orders
python -m cacheproof snapshot save --workspace .cacheproof
python -m cacheproof snapshot restore --workspace .cacheproof
python -m cacheproof ui --workspace .cacheproof
```

完整可重复演示：

```bat
python -m cacheproof demo --workspace demo-workspace --reset --report artifacts\validation-results.json
```

演示依次执行初始化、容量拒绝、写入/删除、视图整体重建、快照重放、子进程强杀恢复、三类异常注入。除真实单调时钟和随机操作 ID 外，容量证明、视图差异与业务码可由相同操作序列复现。

## 声明格式

`samples/cache_spec.json` 同时包含段声明与读视图清单：

```json
{
  "format_version": 1,
  "segments": [
    {
      "segment_id": "seg-alpha",
      "capacity_bytes": 256,
      "entries": [
        {"key": "alpha/a", "size_bytes": 48, "content_seed": "alpha-a-v1"}
      ]
    }
  ],
  "views": [
    {
      "view_id": "view-orders",
      "version": 1,
      "segments": [
        {"segment_id": "seg-alpha", "entries": ["alpha/a"]}
      ]
    }
  ]
}
```

视图快照条目可写成键字符串，也可写成 `{"key": "...", "digest": "sha256:..."}`。同一个 `view_id` 若在清单中出现多次，`version` 必须严格递增；系统初始化时取最高版本。视图可以故意携带过期摘要，用来演示整体重建。任何字段不合法都整体拒绝，返回 `E_SPEC`，并尽可能给出首个违规 `segment_id`、`entry_key` 与 `field`。

## 容量证明与摘要规则

内容生成完全确定：

```text
block(n) = SHA256("CACHEPROOF-CONTENT-v1\0" || len(seed_utf8)_u64be || seed_utf8 || n_u64be)
content   = (block(0) || block(1) || ...)[:size_bytes]
```

条目摘要：

```text
entry_digest = "sha256:" + SHA256(
  "CACHEPROOF-ENTRY-v1\0" ||
  canonical_json({"key": key, "size_bytes": size_bytes}) ||
  "\0" || content
).hexdigest()
```

其中 canonical JSON 为 UTF-8、对象键排序、无多余空白、无结尾换行。内容文件路径为：

```text
entries/<base64url(segment_id)>/<base64url(entry_key)>/<digest_hex>.bin
```

证明包含 `capacity_bytes`、`used_bytes`、`remaining_bytes`、`entry_count`、`segment_set_version`、`segment_version`、按 UTF-8 键序排列的条目摘要清单和 `content_digest`。复核时会重新读取目录内容，不只信任 SQLite 中的证明文本。

## 事务与恢复语义

写入采用三阶段：

1. 单独提交 `prepared` 意图记录（操作 ID、目标路径、期望摘要）。
2. 写临时文件，校验大小与摘要，`fsync` 后原子替换到内容目录。
3. 在一个 SQLite 事务中写条目、更新段证明、段集合版本、完成游标、操作状态和账本内容摘要。

删除先登记 prepared，再在同一证明事务中删除条目并更新容量，提交后移除旧内容文件。若进程在文件替换后、证明事务提交前被强杀，重启只恢复已提交状态，并把 prepared 操作标为 `unknown`，列出条目键、期望摘要和实际观测摘要。未知文件不参与容量证明，因此证明仍可独立通过；它们也不会被冒充为已提交条目。

系统绝不生成“强杀进程退出报告”。恢复报告的字段明确为 `exit_report_available: false`，依据是已提交 SQLite 行和文件系统观测。

## 读视图收敛

视图查询会按稳定顺序比较每个引用段的 `segment_version` 与条目摘要集合。任一不一致即以单条 SQLite 行原子替换视图快照，而不是逐条修补。差异稳定列出：

- `added`
- `deleted`
- `digest_changed`
- 段版本变化

SQLite 使用 WAL；重建事务未提交时，其他连接仍读取旧快照，不会观察到半重建状态。Tkinter 刷新只读，不调用重建、恢复或快照保存，因此刷新不改变账本批次或内容摘要。

## 会话快照与重放

`snapshots/session.json` 包含段集合版本、账本批次、完成游标、最后操作序号、逐段证明、视图快照与未知条目，并带整体内容摘要。恢复时：

1. 校验快照内容摘要与格式版本。
2. 比较快照与当前账本；一致则不做修改。
3. 不一致时，先整体重建过期视图，再按账本操作序号重放快照之后的写入、删除与容量拒绝。
4. 将重放出的段已用字节、条目摘要、视图结果和业务码与账本逐项比较。
5. 全部一致则原子整体替换快照文件，并稳定列出差异；首个不一致返回 `E_REPLAY`，定位到段、条目键和字段。

## 业务码

| 业务码 | 退出码 | 含义 |
|---|---:|---|
| `OK` | 0 | 成功 |
| `E_SPEC` | 2 | 声明或参数非法 |
| `E_CAPACITY` | 3 | 容量不足或容量被改小于已用字节 |
| `E_INTEGRITY` | 4 | 内容文件/证明复核失败 |
| `E_SNAPSHOT` | 5 | 账本、视图或会话快照摘要不符 |
| `E_REPLAY` | 6 | 快照重放与账本不一致 |
| `E_TK_UNAVAILABLE` | 7 | Tkinter/tcl-tk 不可用 |
| `E_IO` | 8 | 文件系统错误 |
| `E_NOT_FOUND` | 9 | 工作区、段、条目或视图不存在 |

## 异常注入

以下命令只改变输入文件或校验参数，并返回对应业务码；结构化 JSON 日志位于工作区 `logs/events.jsonl`，每条包含单调纳秒时间、事件序号和操作序号。

```bat
python -m cacheproof diagnose --workspace .cacheproof --inject truncate --json
python -m cacheproof diagnose --workspace .cacheproof --inject capacity --json
python -m cacheproof diagnose --workspace .cacheproof --inject ledger-digest --json
```

预期依次为 `E_INTEGRITY`、`E_CAPACITY`、`E_SNAPSHOT`。`truncate` 会截断一个已提交内容文件；`capacity` 仅改变复核参数；`ledger-digest` 修改 SQLite 账本中的摘要字段。请只在演示工作区注入。

## Python API

```python
from cacheproof import CacheProofStore, load_spec

spec = load_spec("samples/cache_spec.json")
with CacheProofStore("workspace", spec=spec, create=True) as store:
    report = store.verify_all()
    store.put_entry("seg-alpha", "alpha/c", 40, "alpha-c-v1")
    view = store.get_view("view-orders")
    store.save_snapshot()
    restored = store.restore_snapshot()
```

## 测试与文档

```bat
python -m unittest discover -s tests -v
python scripts\run_validation.py
```

- 技术设计：`docs/TECHNICAL.md`
- 实际验证记录：`docs/VALIDATION.md`
- 验证产物：`artifacts/validation-results.json`

## 边界

- 写入协调使用单进程 `RLock` 加 SQLite `BEGIN IMMEDIATE`；跨进程并发读取安全，跨进程并发写入由 SQLite 串行化，但不提供分布式租约。
- 未知条目会保留审计记录；当前版本不会自动“清除未知”。
- 不使用 Docker、网络服务或系统保留路径。
