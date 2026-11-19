# 实际验证记录

验证日期：2026-11-20（Asia/Shanghai）。

## 环境

- Python：3.11.0rc1（headless Linux）
- SQLite：3.37.2
- NumPy：未安装；核心使用纯标准库回退。SHAKE/SHA 路径在有无 NumPy 时逐字节一致。
- Tkinter：当前环境未安装；`gui.py` 已完成编译检查，需在 Windows 官方 Python 3.10+ 上运行显示验证。
- 未使用 Docker、容器、网络服务或外部账号。

## 自动化测试

命令：

```text
python3 -m unittest discover -s tests -v
```

结果：`Ran 27 tests ... OK`。

覆盖项：

- 规格：重复物理分片、非正字节数、块区间断裂、关系目标缺失、绝对路径、环。非法输入均为 `E_SPEC`，成环附加 `E_CYCLE`。
- 完整性：同内容重复摘要一致；截断、翻转、整体替换、删除均被登记；注入不修改规格源码。
- 传播：`s2 -> s4 -> s5 -> s7`；直接/间接标记；重排节点与关系后轨迹字典序完全一致。
- 增量：仅 `s4/s5/s7` 有任务；`s0/s1/s2/s3/s6` 产物文件字节、摘要和版本不变；与全量重建一致；篡改触发 `E_INCR` 并指出 `s0.bytes`。
- 调度：节点配额 1 时实测峰值为 1；全局槽位 2 时峰值不超过 2；真实超时发生两次并按最终 `timeout_retries_exhausted` 跳过；零配额记录原因。
- 修复：完全修复 3/3；禁用目标产生部分修复；三副本全部损坏时报告不可修复、缺失序号 `3` 与 3 个已尝试副本；三类计数和为块总数。
- 快照：内部摘要篡改触发 `E_SNAPSHOT`；恢复为整体替换且备份权限无可写位；重放差异触发 `E_REPLAY`。
- 崩溃：子进程提交 1 个任务后调用 `os._exit(17)`；恢复报告 1 个已提交、至少 1 个 `unknown_requires_rerun`，`exit_report_available=false`；随后 `--resume` 依据提交记录恢复并补齐 `s4/s5/s7` 产物。
- 异常日志：`E_CYCLE`、`E_INTEGRITY`、`E_SNAPSHOT` 均写入 JSONL，含分片标识、版本和时刻字段。
- 重复演示：同一临时工作区连续运行两次，`stable_digest` 相同。

## 完整演示结果

命令：

```text
python -m shardrep demo --root . --simulate-ms 1
```

关键结果：

- 第一阶段修复分类：`full=3, partial=0, unrecoverable=0`。
- 第二阶段受影响：`s4, s5, s7`；打包任务：`s4, s5, s7`。
- 未受影响校验：`s0/s1/s2/s3/s6` 的版本、内容摘要、产物文件字节均不变。
- 增量与全量：`ok=true`。
- 快照恢复：检测到磁盘漂移，`replaced=true`，旧快照备份权限 `r--r--r--`。
- 快照重放：`ok=true`，比较字段包括 affected、pack manifest、schedule、repair counts/categories、registry digest、block replica digest。
- `stable_digest`：`sha256:618fc2fb74778b40c4a788aa06b2d8cec7a7421f8b4ebc869f2433e5f8f5fe08`。

演示报告、稳定事实、事件日志与 SQLite 账本分别位于 `demo-run/` 下的固定相对路径。报告保留漂移注入记录；演示结束时重新生成教学样本，使 `samples/spec.json` 与节点文件保持干净、可重复。

## 编译检查

```text
python3 -m compileall -q shardrep tests
```

结果：成功，无语法错误。

