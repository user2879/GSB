# 架构与技术说明：文档快照代际迁移与读视图诊断组件

## 1. 模块总览

| 模块 | 职责 |
| --- | --- |
| `snapshot_migration/errors.py` | `E_CHAIN / E_BOUND / E_INHERIT / E_CLEAN / E_CURSOR` 错误码与定位上下文 |
| `snapshot_migration/model.py` | 不可变领域模型：代际、键区间、容量上下限、代际链 |
| `snapshot_migration/chain.py` | JSON 加载与强校验，失败整体拒绝（`E_CHAIN`），不触碰数据库 |
| `snapshot_migration/fragments.py` | 显式拆分/合并规则、规范字节、摘要、片段不变量校验 |
| `snapshot_migration/repository.py` | SQLite schema、片段版本、迁移/清理/事件账本、读引用、整体替换 |
| `snapshot_migration/engine.py` | 代际迁移（继承/重写/新增/废弃）、引用计数、安全清理、快照与差异分类 |
| `snapshot_migration/readview.py` | 双维游标扫描、版本锚定、续读与 `E_CURSOR` 失效 |
| `snapshot_migration/diagnostics.py` | 仅诊断用的良性/破坏性重包，用于演示游标稳定性 |
| `snapshot_migration/recovery.py` | 完整性校验、三类异常注入、从最早未提交代际重放并逐字段比对 |
| `snapshot_migration/demo.py` | 六个确定性场景编排 |
| `snapshot_migration/cli.py` | 命令行入口 |
| `snapshot_migration/ui.py` | Tkinter 诊断控制台（无头环境安全降级） |

## 2. 区间与片段边界规则（显式声明）

- 键区间采用半开边界 `[lower, upper)`；相邻键区间满足 `left.upper == right.lower`，
  因此首尾相接、不重叠且不留空隙。
- 片段规则：
  - **R1 不跨区间**：一个片段只属于一个键区间。
  - **R2 容量**：每个已提交片段满足 `lower <= key_count <= upper`。
  - **R3 超限拆分**：取满足容量的最少片段数 `k`，按尽量均衡的大小在显式拆分点
    （右侧片段首主键）一分为多；无法可行分区时报 `E_BOUND`。
  - **R4 不足合并**：拆分后低于下限的尾部并入前驱；合并后若超上限则按 R3 重新均衡。
  - **R5 相接一致**：首片段下界=区间下界，末片段上界=区间上界，
    片段 `n` 的上界等于片段 `n+1` 的下界（其首主键）。
- 片段主键以规范 JSON 字节存储，摘要为该字节的 SHA-256；继承即逐字节复用，
  摘要相同即证明内容一致。

## 3. 迁移处置判定

- `inherit`：要求与父代际同区间主键逐字节一致，否则 `E_INHERIT`；
  新片段记录 `inherited_from` 与 `source_fragment_id`，版本沿用父版本。
- `rewrite`：按当前容量重新生成片段，账本记录被解除引用的父片段旧版本号。
- `new`：无父区间，直接生成。
- `discard`：以墓碑区间声明（无主键），标记待清理；在仍有祖先读视图时不物理删除。

每个区间的处置类型、复用证据（摘要、来源代际/片段、复用字节）与受影响片段
写入 `migration_ledger`，裂合轨迹写入 `fragment_events`。

## 4. 读视图与双维游标

- 打开读视图即对目标代际及其全部祖先登记引用计数；关闭时逐条释放。
- 扫描计划在视图打开时物化为 `(键区间, 主键, 片段, 版本)` 的有序单元，
  因此后续代际迁移不会改变该视图结果，保证单一代际稳定视图。
- 游标为 `(键区间, 主键, 片段标识, 片段版本)`，续读从锚点的下一行开始，
  不跳行、不重读。
- 诊断重包使片段版本变化时：若锚点主键仍在且顺序可定位，则按主键安全重锚；
  否则抛 `E_CURSOR`，并给出失效片段标识与原主键。

## 5. 引用计数与安全清理

- 废弃区间的受保护集合为“执行废弃的代际及其全部祖先”。
- 仅当该集合与活跃读视图引用无交集时才物理清理，清理前再次校验无活跃引用。
- 清理证据（废弃者、受保护祖先、清理时引用、摘要）写入 `cleanup_ledger`；
  账本对 `(代际, 区间, 片段)` 唯一，重复运行不产生重复记录。

## 6. 快照、整体替换与恢复

- 快照清单包含整数 `format_version`、schema 版本、容量参数版本、代际链摘要、
  全部片段摘要、继承映射、清理账本、代际指针与整体内容摘要。
- 重开时若版本/内容摘要与当前声明不一致，执行**整体替换**（清空派生表后重建），
  绝不增量合并；差异按稳定身份 `(键区间, 序号)` 配合摘要分类为
  新增 / 继承复用 / 重写 / 归属变化 / 移除。
- 重放：在临时库从最早未提交代际起重放迁移与清理，逐字段比较
  下界、上界、主键数、摘要、代际归属、继承来源、状态与清理证据，
  首个不一致精确定位到代际、片段、字段。

## 7. 异常注入与错误码

| 命令行故障 | 错误码 | 定位字段 |
| --- | --- | --- |
| 片段边界/主键数不一致 | `E_BOUND` | `generation_id`、`fragment_id`、`field` |
| 继承来源代际/片段缺失 | `E_INHERIT` | `generation_id`、`fragment_id`、`field` |
| 清理账本与代际指针冲突 | `E_CLEAN` | `generation_id`、`fragment_id`、`field` |
| 链声明非法 | `E_CHAIN` | `generation_id`、`range_id` |
| 游标无法重锚 | `E_CURSOR` | `fragment_id`、`range_id`、`key` |

## 8. Windows 本地运行

前置：Windows 10/11、64 位 Python 3.11/3.12，`python` 与 `pip` 在 PATH；
仅用标准库 `sqlite3`、`tkinter`，无第三方依赖、不访问网络。

```bat
scripts\validate_chain.bat
scripts\run_demo.bat
scripts\run_tests.bat
scripts\migrate.bat
scripts\launch_ui.bat
```

界面需在带图形会话的桌面双击/命令行启动；无显示环境下导入不报错，
仅启动界面时给出明确提示。
