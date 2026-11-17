# 事件总线流式集成系统（本地版）

面向本地环境的事件总线系统，聚焦订阅者状态演进与恢复一致性。仅使用
Python 3.10+ 标准库（`sqlite3`、`json`、`argparse` 等），无第三方依赖，
无需容器，跨平台（Windows / Linux / macOS）均可运行。

## 目录结构

- `eventbus/spec.py` — 事件源 / 订阅者 JSON 加载与强校验（`E_SPEC`）
- `eventbus/machine.py` — 订阅者状态机（idle / processing / committed / failed）
- `eventbus/ledger.py` — SQLite 状态迁移账本、进度表、版本整体替换（`E_LEDGER` / `E_STATE` / `E_PROGRESS`）
- `eventbus/bus.py` — 订阅路由、强制终止恢复、一致性重放（`E_REPLAY`）
- `eventbus/logutil.py` — 结构化 JSON 行日志（含业务码、订阅者标识、事件序号）
- `main.py` — 命令行入口（run / recover / replay / inject）
- `tools/verify.py` — 独立验证工具（不依赖 eventbus 包）
- `samples/` — 确定性样本事件源与订阅者定义
- `demo.py` — 三场景可重复演示脚本
- `tests/test_eventbus.py` — 单元测试（14 例）

## 快速开始

```bash
python demo.py                 # 顺序执行冷启动 / 强杀恢复 / 版本替换三场景
python -m unittest discover -s tests -v
python tools/verify.py --ledger var/ledger.db --sources var/sources.json
```

## 命令行用法

```bash
# 冷启动处理全部事件（恢复 + 重放校验 + 路由处理）
python main.py run --sources samples/sources.json --subscribers samples/subscribers.json --ledger var/ledger.db

# 模拟强制终止：提交 5 个事件后进程以 os._exit(3) 死亡（无退出报告）
python main.py run --kill-after 5 --ledger var/ledger.db

# 仅恢复并输出恢复报告 + 重放校验
python main.py recover --ledger var/ledger.db

# 异常注入（仅修改输入文件，不触碰系统源码）
python main.py inject corrupt-source --sources var/sources.json      # -> E_SPEC
python main.py inject corrupt-ledger --ledger var/ledger.db          # -> E_LEDGER
python main.py inject progress-overflow --ledger var/ledger.db       # -> E_PROGRESS
```

## 关键设计

- **状态机**：允许迁移 `idle->processing`、`processing->committed/failed`、
  `committed->processing`、`failed->processing`；非法迁移以 `E_STATE` 拒绝且
  原状态不变。同一事件在同一订阅者上重复 committed 同样以 `E_STATE` 拒绝。
- **事务对应**：committed 迁移与进度更新在同一 SQLite 事务内写入；启动时
  校验进度不得越过账本已记录的最大 committed 序号，违反报 `E_PROGRESS`。
- **整体替换**：账本 meta 记录 `format_version` 与订阅者集合 `set_version`；
  任一变化即整体替换（旧库复制为只读备份 `*.backup-f<旧格式>-s<旧集合>`，
  新库 meta 记录被替换集合标识与两版本号差异），禁止增量合并。
- **恢复**：仅以 SQLite 中完整写入（`complete=1`）的记录重建内存视图；
  无结果的 processing 迁移被持久标记为未知（`complete=0`）并要求重新处理；
  尾部不完整记录被丢弃。恢复报告列出订阅者数、各订阅者进度、未知迁移数
  与尾部记录处理方式。强制终止时不保证生成任何退出报告。
- **重放校验**：按账本 meta 中持久化的订阅者集合快照与事件源声明的事件
  序列重新执行状态迁移，逐项比较 from/to 状态、进度与迁移次数；首个差异
  定位到订阅者标识、事件序号与字段，报 `E_REPLAY`。重放只读持久化数据。

## 实际验证结果（本机，Python 3.11 + SQLite 3.37）

- 单元测试：14/14 通过（规格校验、状态机、账本、恢复、重放）。
- 场景一 冷启动：22 次提交；进度 sub-audit=8、sub-billing=8、sub-warehouse=7；verify PASS。
- 场景二 强杀恢复：`--kill-after 5` 退出码 3；恢复报告 unknown_transitions=1、
  进度 sub-audit=2 / sub-billing=3 / sub-warehouse=1；补处理 17 次提交后 verify PASS
  （1 条不完整行被忽略）。
- 场景三 版本替换：set_version 1->2 触发整体替换，旧库只读备份
  `ledger.db.backup-f1-s1` 确认只读；替换记录含被替换集合 `subs-core` 与版本差异；
  重建后 verify PASS。
- 异常注入：corrupt-source -> `E_SPEC`（exit 2）、corrupt-ledger -> `E_LEDGER`（exit 5）、
  progress-overflow -> `E_PROGRESS`（exit 4）；篡改迁移前后状态 -> `E_REPLAY`（exit 6）。
  以上均写入 `var/eventbus.log` 结构化日志（含业务码、订阅者标识、事件序号）。
