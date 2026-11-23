# 实际验收记录

执行环境：Linux 开发容器，Python `3.11.0rc1`，SQLite `3.37.2`，NumPy `1.26.4`，Tk `8.6`（本地解包运行库验证）。交付目标为 Windows 本地 Python/Tkinter/SQLite，不依赖容器或网络。

## 自动化测试

命令：

```bash
python -m unittest discover -v
```

实际结果：

```text
Ran 29 tests in 3.633s

OK (skipped=3)
```

通过的覆盖项包括：规格强校验与首个违规路径、环境/来源/观测顺序无关字节一致性、绑定跳过、三种仲裁、误差带稳定决胜、64 位饱和、失效传播与未受影响键字节不变、视图/缩放口径版本递增、单视图事务恢复、SIGKILL 强制终止、快照版本/摘要不一致整体替换、账本重放差异 `E_REPLAY`、三类 CLI 注入、只读刷新不改账本摘要。

默认测试跳过的三项是 GUI 模块导入和两项 GUI 窗口冒烟，因为该容器默认 Python 搜索路径中没有系统 Tk 共享库。随后使用本地解包的 Tk 8.6 库和现有显示服务单独运行这三项，结果均为 `ok`：

```text
tk 8.6
ui_module /home/zsl/feedback/uidfc5909d3-default-166/config_arbiter/ui.py
app_class ArbiterApp
```

同时执行 `python -m compileall -q config_arbiter tests`，结果 `compileall OK`。

在容器已有显示服务上实际构造 `ArbiterApp`，加载已完成账本，检查五类视图内容后自动关闭：

```text
GUI smoke OK
views=3
env_rows=42
chain_rows=73
authority_rows=32
error_rows=13
version_rows=3
status=只读刷新完成 · 账本 92923c44dec4 · 已完成 3 / 3 · 待重跑 0
```

这验证了 Tk 窗口初始化、只读刷新、稳定行渲染以及所有诊断页签的数据装载。

另外在临时目录实际执行 UI 编辑：选中 `prod-cn/http.timeout_ms` 的观测，将取值改为 `3009.125` 并保存，环境 JSON 写回成功，界面状态变为“环境声明已保存；尚未提交裁决”，只读模型把参与视图标记为 `changed`。交付样本未被修改。

空账本场景也已实际打开：3 个声明视图全部显示为 `unknown`，版本页生成 3 条“requires_rerun”当前状态行，工具栏显示“已完成 0 / 3 · 待重跑 3”。

默认路径实际命令链路：

```text
run:     ok  [cost-focus, ops-authority, release-guard]  snapshot 47259da29cef17ae
status:  completed=[cost-focus, ops-authority, release-guard]  unknown=[]
replay:  compared_count=3
restore: mode=replayed  reason=state_and_versions_match  compared_count=3
```

`logs/arbiter.jsonl` 依次记录四个 `command_started`/`command_succeeded` 事件。

## 固定样本裁决结果

`ops-authority` 视图使用缩放 1000，实际键结论：

| 键 | 口径 | 权威定点值 | 权威来源/产物 | 误差影响 | 采纳/否决 |
| --- | --- | ---: | --- | --- | --- |
| `cache.ttl_s` | interval_midpoint | 62250 (`62.25`) | 交集簇中点 | 是 | 采纳 staging+dr，否决 prod 的 59.5 |
| `db.pool_size` | latest | 12000 (`12.0004` 定点截断) | prod-cn/runtime | 是 | 1 采纳 / 2 否决 |
| `edge.latency_ms` | latest | 100000 | prod-cn/runtime | 是 | 定点差 1 落在误差界 2 内 |
| `http.timeout_ms` | latest | 3002004 | prod-cn/override | 是 | dr 名义最新，稳定决胜改选 prod |
| `rate.limit_qps` | weighted_majority | 180000 | 支持度 5:1 | 否 | 采纳 prod+staging，否决 dr |
| `retry.backoff_ms` | weighted_majority | 250000 | 支持度 5:1 | 是 | dr 的 250.05 落在误差带但支持度低 |

该视图 `view_aggregation_error_upper_bound_fixed_units=2001`。

链对照实例：`prod-cn/http.timeout_ms` 的链顺序为 `runtime@90 -> override@60 -> baseline@10`，runtime 因观测序号 80 小于绑定下限 100 被跳过，override 生效；`staging-eu` 的顺序为 `override@70 -> baseline@20`，但绑定只允许 baseline，因此 baseline 生效。两条链签名不同且均写入结论。

## 强制终止恢复

通过 `CONFIG_ARBITER_TEST_CRASH_AFTER_VIEW=1` 在第一个视图事务提交后发送实际 `SIGKILL`：

- 进程非零退出；
- 快照文件不存在，stdout 不含完成报告；
- 重启后账本校验通过，1 个视图 `complete`，2 个视图 `unknown/requires_rerun`；
- 再次正常运行后 3 个视图全部 `complete`。

这证明恢复只依据已提交事务，并且没有声称生成终止瞬间的退出报告。

## 快照与重放

- 一致状态 `restore`：模式 `replayed`，逐项重算/比较 3 个视图。
- 修改环境后 `restore`：模式 `replaced`，快照中的环境文档、观测和结论整体恢复，不增量合并。
- 修改账本中 `ops-authority/http.timeout_ms` 权威定点值并同步伪造账本内容摘要后执行恢复：定位到

```text
code=E_REPLAY
view_id=ops-authority
key=http.timeout_ms
field=authoritative.value_fixed
```

- 修改快照摘要但未重算 `content_digest`：`E_SNAPSHOT`。

## CLI 注入

同一固定样本和账本上实际执行：

```text
E_SPEC      exit 2   $.environments[0].depth
E_REF       exit 3   __injected_missing_environment__
E_SNAPSHOT  exit 4   injected=true
```

环境与视图样本 SHA-256 在注入前后保持不变，`logs/arbiter.jsonl` 中依次出现带时间戳的 `command_failed` 记录。

## 只读刷新

连续两次构建诊断投影后，账本 `content_digest`、视图版本、历史和陈旧记录均不变；两次投影的规范 JSON 字节完全相同。
