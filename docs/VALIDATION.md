# 实际验证记录

## 环境

```json
{"platform":"linux","python":"3.11.0rc1","sqlite3":"3.37.2","tkinter":false,"tkinter_version":null}
```

实际执行环境为 WSL/Linux，回环 TCP、SQLite、CLI、依赖图与 Tkinter 源码语法均已验证。当前容器的 Python 未安装 `tkinter`，因此 GUI 未做窗口级人工点击验证；Tkinter 模块通过条件导入验证，缺少时明确退出。目标 Windows 官方 Python 自带 `tkinter`，Windows 启动脚本会先执行 `check-env`。

## 自动测试

命令：

```bash
python3 -m unittest discover -v
```

结果：`Ran 11 tests in 28.851s`，`OK`。

覆盖点：

- 相同文件/分片大小的逐字节确定性、连续序号与 SHA-256 摘要。
- `E_CHUNK`：非法大小、空文件、错误分片长度。
- 状态迁移、中止保留分片、位图恢复、已提交会话拒绝续传/中止/重复提交。
- 部分上传后的客户端续传只发送缺失分片，已登记分片的摘要与登记单调时钟不变。
- 分片文件损坏时 `E_INTEGRITY`、会话保留、无可见对象、修复后成功提交。
- 重传失效的稳定拓扑顺序、节点版本递增、未受影响节点版本与摘要不变。
- 多版本单调递增、整体替换、旧版本过期、审计、快照整体恢复与删除失效。
- `E_AUDIT` 首偏差对象与分片序号。
- `E_CHUNK`、`E_INTEGRITY`、`E_VERSION` 结构化日志。
- 合并中重启标记 `unknown_needs_rerun`、无可见对象、重跑提交成功。
- 真实 TCP 下的全局并发上限、单会话配额、排队/起止时钟、重复分片收敛。
- 本地 JSON 场景执行。

另以真实服务执行 `python -m chunkdeps.cli script --file samples/demo-scenario.json --json`：11 个步骤全部成功，两次审计均通过。

## 完整演示

命令：

```bash
python3 -m chunkdeps.demo
```

结果：`ok=true`，29 项检查全部通过，耗时 `8219 ms`。完整结构见 `artifacts/validation_report.json`，不含随机会话标识与耗时波动的稳定摘要见 `artifacts/validation_summary.json`。

样本：

- 路径：`artifacts/demo-workbench/windows-sample.bin`
- 大小：`8249` bytes
- SHA-256：`6fc9698d1bd67a6ad003ceada2f471e385854ecb0ced0054d9c7c929ca8edd53`
- 与 `samples/sample_declaration.json` 完全一致。

并发：

- 服务限制：全局 `3`，单会话 `2`。
- 两个会话并行，每个 `17` 片，客户端每会话请求并发 `6`。
- 观测全局峰值 `3`，两个会话峰值均为 `2`，未越过配额。
- 两个对象最终内容摘要都等于样本 SHA-256。
- 调度记录包含 `queue_ms`、`start_mono_ms`、`end_mono_ms`、`duration_ms`。

版本与失效：

- v1 使用 `1024` bytes 分片；v2 使用 `2048` bytes 分片和不同标签，版本从 `1` 单调递增到 `2`。
- 稳定字段差异：`chunk_count`、`chunk_set_digest`、`chunk_size`、`manifest_digest`、`metadata_digest`、`object_path`、`session_id`、`tags`、`version`。
- v1 在 v2 提交后为 `expired`、不可见。
- 控制对象 `archive/other/stable.txt` 的元数据节点和 `archive/other` 前缀视图在 v2 提交前后版本号与摘要逐字节一致。
- 分片重传产生的失效事件包含直接与间接后代，按稳定拓扑顺序排列。
- 中止后恢复跳过 5 个已登记分片，缺失为 0。

快照：

- 保存后删除活动对象，恢复报告 `mode=whole-state-replacement`、`incremental_merge=false`。
- `metadata_version_changes` 与 `chunk_set_changes` 都识别出 `archive/demo/sample.bin`。
- 恢复后审计通过，活动版本回到 v2。

故障与强杀：

- 分片截断注入：`E_CHUNK`。
- 登记摘要损坏：提交返回 `E_INTEGRITY`，会话保持 `uploading` 和完整位图；重传首片后提交成功。
- 竞争版本：另一会话先发布 v1，旧会话以基础版本 0 提交返回 `E_VERSION`。
- 结构化日志计数：`E_CHUNK=2`、`E_INTEGRITY=2`、`E_VERSION=2`，共 293 条事件。
- 合并暂停点对服务进程执行强制终止；重启恢复结果 `unknown_merge_sessions=1`，强制终止前的可见版本数为 0，退出报告声明为不可用。
- 重跑提交后版本为 1，可见版本数为 1。

最终状态：

- 活动对象 `7`，对象版本 `8`，会话 `10`，分片登记 `70`。
- 依赖图 `133` 节点、`141` 边，引用完整且无环，`graph_version=452`。
- 最终一致性审计通过。

## 手工 UI 验证说明

界面入口：

```bat
scripts\run_ui.bat
```

界面包含连接状态、会话/对象/分片计数、上传参数、断点续传、中止、提交、删除、审计、依赖图、快照保存/恢复和恢复分类视图。所有网络操作在后台线程执行，主线程只负责 Tk 控件更新。由于当前容器没有 `tkinter`，本次无法提供窗口截图或人工点击结果；`python3 -m py_compile chunkdeps/ui.py` 与无模块降级路径均已执行。
