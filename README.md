# ChunkDeps Workbench

面向本地 Windows 的对象存储分片上传与元数据依赖诊断台。应用由 Python 标准库实现，不需要 Docker、云账号或任何外部服务；客户端只连接 `127.0.0.1` 上的本地 TCP 存储服务模拟进程。

核心能力：

- 按固定正整数分片大小确定性切分样本，逐片上传并登记摘要。
- SQLite 持久化会话、分片、对象多版本元数据、依赖图、审计结论与调度记录。
- 断点续传、重传覆盖语义、原子提交、对象可见性与强制终止恢复。
- 分片摘要 → 分片清单 → 元数据 → 对象键 → 前缀计数视图的增量失效传播。
- 全局并发上限、单会话并发配额、重复分片收敛与单调时钟调度记录。
- 整体版本替换、只读过期版本、快照比较与全量状态恢复。
- 分片/对象/元数据一致性审计，以及 `E_CHUNK`、`E_INTEGRITY`、`E_AUDIT`、`E_VERSION` 结构化故障。
- Tkinter 诊断台与本地 JSON 场景脚本两种交互方式。

## Windows 准备

1. 安装 Python 3.11 或更高版本，使用 Windows 官方安装包，并保留 `tcl/tk and IDLE` 组件。
2. 在仓库根目录打开 PowerShell 或 CMD，执行环境检查：

```bat
py -3 -m chunkdeps.cli check-env --json
py -3 samples\make_sample.py
```

输出中的 `"tkinter": true` 表示界面可用，`sqlite3` 会显示内置 SQLite 版本。默认只监听 `127.0.0.1:8765`。

3. 启动服务：

```bat
scripts\run_server.bat
```

终端会先输出一行稳定的 `server_ready` JSON，然后常驻。启动前也可检查端口：

```bat
py -3 -m chunkdeps.cli check-port --check-host 127.0.0.1 --check-port 8765 --json
```

4. 另开终端启动界面：

```bat
scripts\run_ui.bat
```

5. 执行完整自动验收：

```bat
scripts\run_demo.bat
```

演示会清空 `artifacts\demo-workbench`，启动真实回环 TCP 进程，执行并发上传、重传失效、版本替换、审计、快照恢复和强制终止恢复，最后生成完整报告 `artifacts\validation_report.json` 与稳定摘要 `artifacts\validation_summary.json`。运行单元测试：

```bat
scripts\run_tests.bat
```

如需重新开始空白工作台，执行 `scripts\reset_workbench.bat`。

## 快速 CLI 流程

所有客户端命令的全局选项 `--host`、`--port`、`--timeout`、`--json` 可写在子命令前后。`--json` 输出稳定排序的单行 JSON，适合脚本与验收。

```bash
python -m chunkdeps.cli split --file samples/windows-sample.bin --chunk-size 1024 --output-dir split-out --json
python -m chunkdeps.cli put --key archive/demo/sample.bin --file samples/windows-sample.bin --chunk-size 1024 --tags '{"team":"storage"}' --concurrency 4 --json
python -m chunkdeps.cli audit --json
python -m chunkdeps.cli views --prefix archive --json
python -m chunkdeps.cli graph --events --json
python -m chunkdeps.cli snapshot save --json
python -m chunkdeps.cli snapshot restore --json
python -m chunkdeps.cli recover --json
```

分段操作：

```bash
python -m chunkdeps.cli create --key archive/demo/sample.bin --file samples/windows-sample.bin --chunk-size 1024 --json
python -m chunkdeps.cli upload --session s-... --file samples/windows-sample.bin --concurrency 4 --json
python -m chunkdeps.cli abort --session s-... --json
python -m chunkdeps.cli resume --session s-... --json
python -m chunkdeps.cli commit --session s-... --json
```

## JSON 场景脚本

`samples/demo-scenario.json` 是可直接运行的本地声明，支持 `upload_file`、`create_session`、`upload_session`、`reupload_chunk`、`abort`、`resume`、`commit`、`audit`、`snapshot_save`、`snapshot_restore`、`delete`、`views`、`graph` 等步骤，以及 `$last_session`、`$last_object` 变量。

```bash
python -m chunkdeps.cli script --file samples/demo-scenario.json --json
```

## 业务语义

### 确定性与分片登记

- 分片大小必须是正整数；空文件、非法大小、错误索引或与声明长度不一致的分片返回 `E_CHUNK`。
- 分片序号从 0 连续编号，摘要使用会话声明的 `sha256`、`sha384` 或 `sha512`。
- 相同文件和相同分片大小重复切分得到逐字节相同的分片与摘要。
- `chunks` 表以 `(session_id, chunk_index)` 为主键；重传使用覆盖语义，只保留最新登记，并触发依赖失效。

### 会话状态机

状态包括 `created`、`uploading`、`verifying`、`committed`、`aborted`。每次变化都写入 `session_events`，包括位图、原因和 `CLOCK_MONOTONIC` 毫秒。已提交会话不可续传、重传或中止。被中止会话保留全部分片，恢复时按持久化位图跳过已登记分片。

### 原子提交与完整性

提交前逐片比较磁盘文件与 SQLite 登记摘要，再计算对象内容摘要、清单摘要和分片集合摘要。任意偏差返回 `E_INTEGRITY`，会话回到 `uploading` 以供修复和重试，不生成可见对象。全部校验通过后，对象数据先写同目录临时文件并 `fsync`，在同一个 SQLite 事务中发布元数据、完成标记和版本号，最后原子替换对象文件。进程在合并中终止时，只可能留下不可见临时文件；重启不会把半对象标记为可见。

### 依赖图与增量失效

有向边方向为“依赖 → 被依赖项”。典型链路：

```text
chunk → manifest → stage-metadata → session-summary
manifest → object-metadata → object-key → prefix-view
session-state → manifest
```

每次新增/覆盖分片、删除对象、替换版本或中止会话，都从触发源计算可达后代，按稳定拓扑顺序递增 `entry_version`、清空缓存摘要并置 `valid=0`。事件同时记录触发源、直接/间接层级、前后版本、前后摘要和全局图版本。未受影响节点不改动。所有加密摘要都采用带域分离的 canonical JSON 口径。

### 多版本、审计与快照

- 同一对象键的提交版本号由历史最大值单调递增，旧活动版本变为 `expired` 且不可见；对象文件保留为只读历史。
- 提交新版本整体替换元数据，不使用旧标签或旧清单做增量合并；结果输出稳定字段差异、前后摘要和被替换视图快照。
- `audit` 校验当前版本的分片序列、文件集合、逐片摘要、清单摘要、分片集合摘要、对象内容摘要、元数据摘要、完成标记和版本链接；首个偏差由 `E_AUDIT` 报告。
- 快照包含会话、事件、分片登记、对象版本、依赖图版本、最近审计结论和文件清单。恢复前比较对象版本与分片集合摘要，然后事务性整体替换当前状态；不增量合并。

### 并发与资源控制

服务端以条件变量强制全局并发上限和单会话配额。分片请求超出限制时排队；每个成功请求返回 `queue_ms`、`start_mono_ms`、`end_mono_ms`、持续时间和限制值。同一分片的并发重复请求由服务端锁和唯一主键收敛为一个持久化结果。最终清单始终按分片序号排序，因此网络交错不会改变对象内容摘要或清单顺序。

## 故障注入

三类故障都走正常业务路径并写入结构化 JSONL 日志：

```bash
# 输入分片被截断：E_CHUNK
python -m chunkdeps.cli upload --session s-... --file samples/windows-sample.bin --inject truncate-chunk --json

# 登记摘要被破坏：提交校验发现 E_INTEGRITY，会话保留
python -m chunkdeps.cli commit --session s-... --inject corrupt-metadata --json

# 显式注入版本竞争：E_VERSION，会话保留为 uploading
python -m chunkdeps.cli commit --session s-... --inject version-conflict --json

# 也可复现真实竞争：另一个会话先发布 v1，旧会话仍以 expected_base_version=0 提交
python -m chunkdeps.cli commit --session s-old --expected-base-version 0 --json
```

强制终止测试钩子为环境变量 `CHUNKDEPS_MERGE_PAUSE_MS`。演示在合并暂停点对服务进程执行强制终止，重启后确认会话被标记为 `unknown_needs_rerun`，未生成可见对象、不提供退出报告，并要求重跑提交。

## 错误码

| 业务码 | 含义 |
|---|---|
| `E_CHUNK` | 分片大小/索引/长度/内容不合法或分片文件截断 |
| `E_STATE` | 非法状态迁移、已提交会话再操作 |
| `E_INTEGRITY` | 分片或对象摘要不一致、禁止合并不完整数据 |
| `E_VERSION` | 期望基础版本与当前活动版本冲突 |
| `E_AUDIT` | 一致性审计失败，附带首个偏差对象与分片 |
| `E_CONFLICT` | 本地服务不可达、端口占用或协议连接失败 |
| `E_INPUT` | 参数、对象键、快照或场景声明非法 |
| `E_NOT_FOUND` | 会话、对象或快照不存在 |

## 目录

```text
chunkdeps/       引擎、图、SQLite 账本、TCP 服务、客户端、CLI、Tkinter UI、演示
samples/         确定性样本、声明和 JSON 场景
scripts/         Windows 启动、演示、测试和清理脚本
tests/           引擎与真实 TCP 端到端测试
docs/            设计、协议与验收记录
artifacts/       validation_report.json 等运行产物
```

详细设计见 `docs/DESIGN.md`，协议见 `docs/PROTOCOL.md`，逐条需求追踪见 `docs/REQUIREMENTS.md`，本次实际验证见 `docs/VALIDATION.md`、`artifacts/validation_report.json` 与 `artifacts/validation_summary.json`。
