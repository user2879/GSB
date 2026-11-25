# 验证记录

验证环境：Linux 沙箱，Python 3.11.0rc1，NumPy 1.26.4。目标交付环境为
Windows 官方 Python 3.11/3.12；代码仅使用跨平台标准库 API 与 NumPy。

## 自动化测试

命令：

```sh
python -m unittest discover -s tests
```

结果：32 个测试全部通过，覆盖：

- JSON/NDJSON 加载与 `E_SPEC` 强校验（唯一 ID、回环字面量、端口唯一、
  正整数、修订父链、依赖环、取消前置下发）。
- 确定性样本与 int64 定点聚合的跨运行字节一致性。
- 租约授予、真实单调时钟到期、乐观版本不符 `E_LEASE`、续约上限。
- 稳定拓扑序失效传播与仅相关子图失效。
- `E_HANDSHAKE`、`E_FIELD`、端口与套接字超时分类。
- SQLite 格式版本、内容摘要、篡改后整体替换、unknown 标记。
- 真实子进程端到端运行、预算/租约/并发约束、取消与 blocked 终态。
- 节点强制 kill 后重启握手、unknown 分片重跑恢复。
- Tk 无关的只读 view model 稳定渲染。

## 可重复演示

命令：

```sh
python examples/demo_run.py all
```

结果（见 `demo/demo_run.log`）：
最新完整日志见 `demo/final-demo.log`，单次运行与重放结果见
`demo/single-run.json` 和 `demo/single-verify.json`。

- `full=true`：完整事件流运行后独立重放 `status=ok`，资源占用归零。
- `crash=true`：节点强制终止并重启；unknown 的 `s2` 重跑后 completed，
  unknown 列表为空。
- `tamper=true`：篡改 content digest 后打开账本触发整体替换，
  替换后租约事件数为 0。

## Tkinter 说明

当前沙箱导入 Tkinter 时缺少系统原生库 `libtk8.6.so`，因此未实际弹窗。
`shardflow/diagnostics.py` 延迟导入 Tkinter，无头环境下其余 CLI、测试与
view model 全部可用；Windows 官方 Python 安装包内置 Tkinter，可直接运行
`shardflow diagnose`。
