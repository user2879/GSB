# 本地文档范围二级索引组件

面向 Windows 本地 Python 3.10+ 的确定性、分段式数值范围索引。组件从 JSON 声明加载文档集合，在本地 SQLite 中维护逐段独立表，并支持原子批量变更、陈旧键账本、版本一致性拒绝、稳定分页查询、整库快照替换、审计重放和 Tkinter 诊断界面。

仅使用 Python 标准库：`json`、`sqlite3`、`hashlib`、`tkinter` 等。没有第三方依赖，不访问网络。

## 快速开始（Windows）

在项目根目录执行：

```powershell
python --version
python samples\generate.py --scale 48 --seed 20261120 --capacity 5 --output-dir samples

python -m docdb_index init --session sessions\demo --input samples\collections.json --reset
python -m docdb_index apply --session sessions\demo --input samples\changes.json --batch-id incremental --batch-size 2
python -m docdb_index query --session sessions\demo --collection orders --index metrics.amount --range 0 100 --limit 10
python -m docdb_index scan-full --session sessions\demo --collection orders --index metrics.amount --range 0 100 --limit 10
python -m docdb_index diagnose --session sessions\demo --collection orders --index metrics.amount
python -m docdb_index verify --session sessions\demo
```

一键执行全部演示：

```powershell
.\scripts\demo.ps1
# 或
.\scripts\demo.bat
```

Linux/macOS 可用 `python3` 替代 `python`，并运行 `python3 scripts/demo.py`。

Tkinter 检查：

```powershell
python -c "import tkinter as tk; print(tk.TkVersion)"
python -m docdb_index gui --session sessions\demo --collection orders --index metrics.amount
```

## 声明格式

```json
{
  "collections": [
    {
      "name": "orders",
      "version": 7,
      "primary_key": "id",
      "documents": [
        {
          "id": "doc-000001",
          "customer": {"region": "east"},
          "metrics": {"amount": 123, "latency_ms": 42}
        }
      ],
      "indexes": [
        {
          "path": "metrics.amount",
          "segment_capacity": 5,
          "allow_null": false
        }
      ]
    }
  ]
}
```

当前版本每个集合支持一个索引声明。字段路径由字母、数字、下划线组成，以点分隔。`allow_null=true` 时缺失或 `null` 的索引字段不进入索引；`false` 时加载直接返回 `E_DOC`。主键必须是集合内唯一的非空字符串或有限数值。

## 命令

| 命令 | 作用 |
|---|---|
| `init --session DIR --input FILE [--reset]` | 全量强校验后构建新会话；先写临时库，成功后原子替换 |
| `apply --session DIR --input FILE [--collection C] [--batch-id ID] [--batch-size N] [--retain-stale]` | 原子应用一个变更批次；`--batch-size` 仅改变内部分组，不改变结果 |
| `reclaim --session DIR --collection C` | 回收尚未清理的物理陈旧条目，不改变集合/索引逻辑版本 |
| `query ... --eq V` / `--range LO HI` / `--lo LO --hi HI` | 范围、等值或全序扫描；`--offset`、`--limit` 跨段分页 |
| `scan-full ...` | 从当前文档全量重算的参考查询 |
| `diagnose ...` | 稳定输出分段边界、版本、陈旧条目、过滤计数和账本 |
| `verify ...` | 校验段摘要/边界/账本，并从头重放审计批次 |
| `replay ...` | 仅执行独立重放比较 |
| `inject ... --type segment\|bound\|ledger` | 注入摘要损坏、边界/条目数不一致、账本缺失 |
| `snapshot create/list/restore` | 整库快照创建、列出与整体替换恢复 |
| `manifest ...` | 输出确定性状态清单与全部段摘要 |
| `gui ...` | 启动 Tkinter 诊断界面 |

所有成功/失败结果都输出单行规范 JSON。业务错误退出码为 `2`，成功为 `0`。

## 业务码

| 业务码 | 含义 |
|---|---|
| `E_DOC` | 声明、主键、字段路径、数值或变更不合法；加载不留下部分状态 |
| `E_SEGMENT` | 段内容摘要或内容 BLOB 校验失败 |
| `E_BOUND` | 段边界、条目数、条目顺序或段范围不一致 |
| `E_LEDGER` | 陈旧键账本缺失、多余或摘要不符 |
| `E_REPLAY` | 独立重放与当前段/未受影响段/台账不一致 |
| `E_STALE` | 索引版本与集合版本不一致，拒绝查询或应用 |
| `E_STATE` | 缺少会话、重复初始化等状态错误 |
| `E_SNAPSHOT` | 快照损坏、目录冲突或恢复后摘要不符 |

结构化日志写入 `sessions/<name>/logs/events.jsonl`。

## 核心保证

- 初始构建：条目按 `(数值键, 主键)` 排序后确定性分块；相同输入产生相同段标识、内容 BLOB、摘要和状态清单。
- 增量重建：只重排受影响段；未受影响段的条目、版本、内容 BLOB 与摘要保持不变，并逐段校验。
- 分裂：超过容量时确定性平衡分裂；首个分块保留原段标识，新增分块使用持久化的单调段号。
- 停滞窗口：可用 `--retain-stale` 保留物理陈旧引用；查询按当前文档状态过滤，结果与全量扫描逐字节一致。
- 陈旧账本：区分 `document_delete` 与 `key_migration`，同一 `(主键, 陈旧键)` 幂等；历史账本在物理回收后仍保留。
- 快照：包含集合/索引版本、所有段摘要、账本、查询历史摘要、数据库摘要与整体内容摘要；恢复是整库替换，并保留只读旧快照和恢复前回滚副本。

## 项目结构

```text
docdb_index/          组件实现与 CLI
samples/              确定性生成器、集合、变更批次与参数模板
tests/                unittest 自动化测试
docs/DESIGN.md        结构与算法说明
docs/VERIFICATION.md  实际验证记录
scripts/              Windows/Linux 一键演示
```

运行测试：

```powershell
python -m unittest discover -s tests -v
```
