# 列式存储压缩与扫描诊断工作台（colbench）

面向本地 Windows 环境的列式段存储、压缩编码、谓词下推扫描与并行调度诊断工作台。
仅依赖 **Python 3.11+ 标准库 + NumPy**，无外部服务、无网络访问、无容器。

## 1. 环境准备

```powershell
# Windows：官方 Python 3.11+ 安装包默认包含 tkinter 与 sqlite3
py -m pip install -r requirements.txt
py -m unittest discover -s tests
py demo.py
```

- `tkinter`：Windows 官方安装包默认包含；Linux 无显示/无 tkinter 时
  GUI 自动降级为顺序一致的文本渲染（`colbench/gui.py`）。
- `sqlite3`：标准库；`numpy>=1.26` 用于样本构造与编解码校验。

## 2. 目录结构

| 模块 | 职责 |
| --- | --- |
| `colbench/errors.py` | 统一业务码：`E_SCHEMA/E_PAYLOAD/E_ENCODING/E_BUDGET/E_DIVERGE/E_SNAPSHOT` |
| `colbench/codecs.py` | RAW/DICT/RLE 长度前缀编解码、zlib 压缩、SHA-256 摘要 |
| `colbench/schema.py` | 段定义/块表 JSON 加载与强校验（列、行组、编码、字节数） |
| `colbench/segment.py` | `*.cseg` 段文件落盘/读取、块头（编码/行数/前后字节数/区域图）、块表 JSON |
| `colbench/planner.py` | 查询解析、规范化、列裁剪、等值/范围/IN/AND 谓词下推与计划摘要 |
| `colbench/scheduler.py` | 全局并行度 + 每列内存 + 每段配额三重预算的块级并行调度 |
| `colbench/ledger.py` | SQLite 块级扫描账本，逐批写入、陈旧标记、整体替换与备份表 |
| `colbench/replay.py` | 权威重算（独立路径）与账本逐字段比对，`E_DIVERGE` 定位首分歧 |
| `colbench/versioning.py` | budget/encoding/format 三类单调版本、陈旧防护、整体替换视图 |
| `colbench/snapshot.py` | 会话快照、内容摘要、版本识别、整体替换恢复与只读备份 |
| `colbench/inject.py` | 载荷截断 / 编码不匹配 / 内存预算为零三类异常注入 |
| `colbench/view.py`、`gui.py` | 稳定顺序视图模型；Tkinter 界面 + 无头文本降级 |
| `colbench/sample_data.py` | NumPy 确定性样本：两段 × 3 行组 × 4 列，混合三种编码 |
| `demo.py` | 三场景可重复演示；`verify_numpy.py` NumPy 编解码校验 |

## 3. 关键设计约定

- **确定性字节**：int64 小端、字符串 UTF-8、全部长度前缀 4 字节小端；
  zlib 固定级别；字典项显式排序（整数数值序、字符串 UTF-8 字节序），
  不依赖映射遍历顺序。重复构建段文件 SHA-256 完全一致。
- **段文件布局**：魔数 `CSEG` + 版本 + 列定义 + 行组区间 + 块（块头 +
  长度前缀载荷）+ 结束标记 `CEND`。块头记录编码、整数行数、压缩前/后
  字节数、摘要与 min/max/不同值数。
- **谓词下推**：合取项稳定排序后按块区域图/字典独立判定，输出候选块、
  跳过块与推导依据；`PUSHED` 与 `RETAINED` 分离标注；空白/括号差异得到
  同一 `plan_digest`。
- **三重预算**：所有准入判断在同一把调度锁内完成，先记账后运行；
  排队开始/结束、运行开始/结束使用 `time.monotonic_ns()` 毫秒；
  结束后再次断言不变量，任何越界抛 `E_BUDGET`。
- **账本与重算分离**：账本只记录真实调度轨迹；`replay.py` 直接读段文件
  重放，逐字段比较并定位首个不一致块的列与字段。
- **整体替换**：参数/编码/格式变化使版本单调递增，旧轨迹标 `stale=1`
  不参与汇总；视图 `ViewStore.replace` 清空并列出受影响块与清空条数；
  快照不匹配时旧文件保留为 `*.backup.json` 只读备份。

## 4. 命令行

```powershell
py -m colbench.cli init --workspace data/demo
py -m colbench.cli scan --workspace data/demo `
   --query data/demo/samples/q_range.json --parallelism 2 --memory 2048 --quota 1 --text
py -m colbench.cli verify --workspace data/demo --query data/demo/samples/q_range.json
py -m colbench.cli gui --workspace data/demo          # 有 tkinter 时开窗
```

## 5. 演示场景（`demo.py`，稳定顺序输出）

1. 首次全量扫描：宽松预算（并行 4、内存 8192B、段配额 2），24 个块任务，
   权威重算 `MATCH`。
2. 预算收紧重调度：并行 1、内存 1024B、段配额 1，budget 版本 1→2，
   旧 24 条轨迹标记陈旧并整体替换；附带括号/空白计划摘要等价断言。
3. 快照恢复重放：从段文件重新打开并按当前版本恢复，重算仍 `MATCH`；
   版本/摘要不一致时整体替换并保留只读备份。
4. 异常注入业务码：`truncate→E_PAYLOAD`、`encoding_mismatch→E_ENCODING`、
   `zero_memory→E_BUDGET`。

## 6. 实际验证结果（本机记录）

- `python -m unittest discover -s tests`：**33 个用例全部 OK**，覆盖编解码
  往返/截断/摘要、schema 强校验、裁剪下推、三重预算、账本重算分歧定位、
  版本演进、快照摘要与整体替换、端到端业务码。
- `python verify_numpy.py`：9 组 NumPy 样本 × 三种编码往返与确定性全部通过。
- `python demo.py`：三场景成功，异常码依次为 `E_PAYLOAD/E_ENCODING/E_BUDGET`，
  段文件重复构建 SHA-256 一致，快照备份文件为只读。

## 7. 持久化产物

```
data/demo/
├── samples/            # 确定性段定义 JSON 与查询 JSON
├── segments/*.cseg     # 列式段二进制
├── segments/*.blocks.json  # 块表（块头、字节数、摘要、区域图）
├── ledger.sqlite3      # scan_ledger + scan_ledger_backup
├── session.snapshot.json / *.backup.json
└── versions.json       # 三类单调版本与参数签名
```
