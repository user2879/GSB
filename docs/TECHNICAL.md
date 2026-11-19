# 页级双写缓冲恢复诊断组件 — 技术说明

面向 Windows 本地环境、纯 CPython 3.11+ 标准库实现（`sqlite3`/`tkinter`/`hashlib`/`json`/`struct`），
无第三方依赖、无容器、无外部端口与服务。

## 1. 设计总览

以固定大小页（默认 1024 字节，支持 256/512/1024/2048/4096）为持久化单元。
用户键值变更先按键散列编译为若干页镜像变更，写入双写缓冲并 `fsync` 落定后，
才覆盖主数据文件对应页，最后写完成标记。任一时刻的半写页都能从缓冲取回完整镜像。

模块职责：

| 模块 | 职责 |
| --- | --- |
| `dwb_diag/paging.py` | 页帧编解码、主数据文件定长寻址、结构与摘要判定 |
| `dwb_diag/buffer.py` | 双写缓冲镜像编码、追加、严格/容错扫描、E_PAGE 定位 |
| `dwb_diag/storage.py` | 三阶段写路径、崩溃点标记、崩溃注入 |
| `dwb_diag/recovery.py` | 逐页选镜像、修复、E_LOST、确定性恢复报告 |
| `dwb_diag/views.py` | 两类视图 + rollup、依赖图、环检测、拓扑失效与重建校验 |
| `dwb_diag/snapshot.py` | SQLite 会话快照、版本识别、整体替换、只读备份 |
| `dwb_diag/fault_injection.py` | 四类异常注入、主文件损坏校验、E_REPLAY 重放比对 |
| `dwb_diag/engine.py` | 门面编排、确定性样本、崩溃态备份与重放 |
| `dwb_diag/cli.py` | 命令行入口 |
| `dwb_diag/gui.py` / `gui_model.py` | Tkinter 界面与无 Tk 依赖的只读模型 |

## 2. 文件与记录格式

### 页帧（主文件槽位与缓冲镜像共用同一份字节）

```
magic(4)="PG01" | payload_len(I) | payload(...) | 0x00 填充至 page_size
页内容摘要 = sha256(整帧)
```

### 双写缓冲镜像（追加，定长头 56 字节 + 页帧）

```
magic(4)="DWBM" | page_no(I) | page_size(I) | seq(Q) | length(I) | digest(32s) | frame(length)
```

解码严格校验：魔数/截断、页号在容量内、页大小受支持、长度匹配、摘要匹配、
写入序号在整个缓冲内严格递增。任一非法以 `E_PAGE` 整体拒绝，并稳定输出
首个出错镜像的缓冲偏移 `offset` 与页号 `page_no`（含 `reason`）。

### 会话目录布局（`DataLayout`）

- `doublewrite.dwb` 双写缓冲；`main.pages` 主数据文件（页号 × page_size 定长寻址）
- `page_versions.json` 页版本映射（页号→最新序号+权威摘要）
- `views.json` 视图定义；`crash_markers.jsonl` 三阶段崩溃点标记
- `sessions.sqlite` 恢复会话快照；`events.log.jsonl` 结构化日志
- `pre_recovery/` 恢复前崩溃态备份，供确定性重放

## 3. 写路径三阶段与崩溃判定

每次提交固定顺序，且每阶段先写崩溃点标记（序号、阶段名、单调时钟毫秒）：

1. `buffer_write`：全部页镜像追加进 DWB 并 `fsync`
2. `main_overwrite`：逐页覆盖主文件槽位并 `fsync`（可在中途崩溃形成半写）
3. `commit_mark`：页版本映射落盘并写完成标记

恢复时依据最后标记的阶段判定半写位置；不存在延后写或省略标记的路径。

## 4. 恢复与权威判定

1. 容错扫描缓冲：结构错误仍 `E_PAGE`；摘要坏镜像计入“丢弃不可用镜像”。
2. 按页取摘要匹配且序号最新的镜像，与主文件槽位摘要逐页比对。
3. 主文件缺失/结构坏/摘要不符即用缓冲完整镜像修复；版本映射同步到缓冲真值。
4. 主文件与缓冲均无可用镜像时抛 `E_LOST`（页号 + 最大序号），绝不零填充冒充。
5. 全部完成后才置 `keyspace.ready=True` 开放键空间作答。

报告含：已修复页数、无需修复页数、被丢弃镜像数、最大写入序号、最终页摘要集合，
并声明真值来源为“缓冲最新可用镜像与主文件摘要的逐页比对”。同一状态重复恢复，
报告与摘要集合逐字节一致（测试 `test_deterministic_repeated_recovery` 验证）。

## 5. 视图依赖与失效传播

- `prefix_count`：依赖前缀键所在页集合；`value_range`：依赖值落在区间内的页集合；
  `rollup`：视图对视图聚合，用于间接传播与成环。
- 页被覆盖或修复时，先收集到集合（与遍历顺序无关），再沿 page→view→view 闭包传播，
  按 Kahn 稳定拓扑序输出失效视图名并递增版本号。
- 依赖成环抛 `E_DEPEND` 并输出环路径。
- 失效后重建只依据当前已恢复主文件与视图定义，不复用旧缓存；
  `verify` 比较版本与重算摘要，版本未递增或摘要异常以 `E_VIEW` 报首个视图名与字段。

## 6. 快照版本识别与整体替换

SQLite 表 `sessions` 保存缓冲摘要、主文件摘要集合、页版本映射、视图版本、
修复轨迹与失效轨迹。重开时比较快照缓冲摘要与主文件摘要：

- 一致：载入内存状态；
- 不一致：记 `E_SNAPSHOT`，旧库复制为应用层只读备份 `*.bak`，
  内存页缓存/版本映射/视图状态**整体替换**而非增量合并，并稳定列出变化页与视图。

## 7. 业务码链路

| 注入 | 业务码 | 触发点 |
| --- | --- | --- |
| 主文件页内容损坏 | `E_CORRUPT` | `assert_main_intact` 对照版本映射摘要 |
| 缓冲镜像截断 | `E_PAGE` | 缓冲严格扫描的截断/序号/摘要校验 |
| 依赖成环 | `E_DEPEND` | 失效传播前的环检测 |
| 快照缓冲摘要不符 | `E_SNAPSHOT` | 快照重开的摘要比对 |
| 重放轨迹/摘要差异 | `E_REPLAY` | 重放逐项比对，定位首个不一致页号与字段 |
| 页无任何可用镜像 | `E_LOST` | 恢复阶段 |
| 视图版本/摘要异常 | `E_VIEW` | 视图校验 |

所有业务码写入 `events.log.jsonl`，四类注入各自独立目录、互不掩盖。

## 8. 验证结果（本机实测）

- `python run_tests.py`：31 个用例全部通过（unittest）。
- CLI `demo`：3 批确定性写入，第 3 批 `main_overwrite` 阶段注入崩溃，
  恢复修复 7 页、17 页完好、丢弃 0、最大序号 24，快照 `s1` 重放 `replay_ok`。
- 四类注入业务码实测：`E_CORRUPT / E_PAGE / E_DEPEND / E_SNAPSHOT` 均正确入日志。
- `replay --tamper-page 5`：`E_REPLAY` 定位 `page_no=5, field=digest`。
- 重复恢复：两份独立副本报告与最终页摘要集合完全一致。
- 当前 Linux 沙箱无 tkinter，`gui.py` 防御性导入不报模块错误，`gui_model` 无 Tk 可测；
  Windows 官方 CPython 自带 Tcl/Tk，可直接 `launch()`。

## 9. Windows 操作

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_demo_windows.ps1
```

常用命令：

```
.venv\Scripts\python.exe -m dwb_diag.cli init       --root data\s1
.venv\Scripts\python.exe -m dwb_diag.cli write      --root data\s1 --kv user:1=10
.venv\Scripts\python.exe -m dwb_diag.cli crash      --root data\s1 --commit 2 --phase main_overwrite --kv user:2=2
.venv\Scripts\python.exe -m dwb_diag.cli recover    --root data\s1
.venv\Scripts\python.exe -m dwb_diag.cli view-verify --root data\s1
.venv\Scripts\python.exe -m dwb_diag.cli inject     --root data\s1 --kind buffer_truncate
.venv\Scripts\python.exe -m dwb_diag.cli replay     --root data\s1 --tamper-page 5
```

## 10. 关键设计取舍

- 页帧与缓冲镜像共用字节，主文件摘要与缓冲摘要可直接逐页比对，避免双重摘要口径。
- 损坏页在只读依赖图构建时容错为空，真正修复统一由恢复引擎依据缓冲完成，
  保证“先恢复后作答”的闸门不被界面/视图路径绕过。
- 重放基于恢复前崩溃态备份，确保重新执行真实修复而非读到已修复主文件。
- 备份采用应用层只读语义而非强制 `chmod 0o444`，避免 SQLite 重建 journal 时打开失败
  （Windows 上尤其明显），并在结构化日志记录备份动作。
