# 实际验证记录

验证日期：2026-11-21（Asia/Shanghai）

执行环境：

- WSL Ubuntu 22.04，本地回环网络，无容器。
- Python 3.11.0rc1。
- SQLite 3.37.2。
- NumPy 1.26.4（仅用于样本生成）。
- 当前 Python 未安装 Tkinter，因此本次实际运行验证了工作台的无界面视图模型、排序与只读刷新；Tk 窗口本身需在勾选 Tcl/Tk 的官方 Windows Python 3.11/3.12 上运行。

## 自动化测试

命令：

```bash
python -m unittest discover -s tests -v
```

结果：

```text
Ran 23 tests in 4.822s
OK
```

主要覆盖：

- 规格路径重复、TTL、缓存键字段重复、缺失依赖、环、内容截断、首错稳定性。
- TTL 递减与过期、条目版本递增、三种逐出规则的访问顺序无关集合、陈旧账本拒绝。
- 内容数组重排后的传播轨迹字节一致。
- 单调账本版本、当前统计仅含最新版本、历史版本陈旧、待定轮次未知化、混合内容版本拒绝。
- 真实套接字握手超时、端口冲突 `E_PORT`。
- 三节点冷启动、内容演进、快照双保存与只读备份、`E_REPLAY`、`E_SNAPSHOT`、整体恢复、无界面视图两次刷新一致。
- 节点/内容源强制终止后无退出报告、恢复、待定轮次未知化、使用持久化账本重新启动。
- 策略版本演进及节点集合扩容（新增 `edge-d`）生成新账本轮次。
- CLI 三类异常注入分别返回 `E_CYCLE`、`E_CONTENT`、`E_SNAPSHOT`。

## 三场景演示

命令：

```bash
python scripts/run_demo.py --workspace /tmp/edgecache-final-demo-20261121b --port-base 43400
```

实际退出码：`0`

关键结果：

- 冷启动账本版本：`1`，内容版本：`cv-0e66881fc1359571`，策略版本：`policy-v1`。
- 节点：`edge-a`、`edge-b`、`edge-c`；初始容量逐出事件按稳定排序记录。
- `/data/root` 更新后的受影响集合：`["/data/left", "/data/right", "/page/home"]`。
- 传播轨迹深度：`1, 1, 2`；所有轨迹行只引用单一内容版本 `cv-98f8cc99b75bc45a` 与单一策略版本 `policy-v1`。
- 传播轨迹摘要：`5b7b226166666563`。
- 快照来自账本版本 `2`；磁盘后来演进到账本版本 `3`；恢复创建账本版本 `4`。
- 快照版本集合比较：`requires_replace=true`。
- 重放校验：`ok=true`、`checked_rows=3`。
- 恢复后当前账本版本 `4`，陈旧版本 `[1, 2, 3]`，当前条目全部属于恢复后的单一内容版本。
- 当前轮单内容版本检查：`true`；单策略版本检查：`true`。

报告文件：`/tmp/edgecache-final-demo-20261121b/report.json`（本次临时路径）。

## 异常注入

实际输出：

| 注入 | 退出码 | 业务码 | 定位 |
| --- | --- | --- | --- |
| `content_truncation` | 2 | `E_CONTENT` | `path=/data/root`, `field_path=content[/data/root].content_b64` |
| `cycle` | 2 | `E_CYCLE` | `path=/data/left`, `field_path=content[/data/left].derived_from`, `spec_rejection=E_SPEC` |
| `snapshot_digest_mismatch` | 2 | `E_SNAPSHOT` | `field_path=content_digest` |

另外实际执行 `python -m edgecache.cli gui --db ...`，当前缺少 Tkinter 的环境稳定返回 `E_TK` 和安装提示；无界面视图模型由自动化测试与 `view` 命令验证。

## Windows 注意事项

本次容器运行环境不是 Windows，也没有 Tkinter。代码路径已按 Windows 处理：

- 子进程使用 `sys.executable`、`CREATE_NEW_PROCESS_GROUP`、`CREATE_NO_WINDOW`，不经过 shell。
- 端口预检使用 Windows 独占绑定；工作进程允许重启时复用自身 `TIME_WAIT` 端口。
- 快照只读使用 Python `chmod` 的 Windows 只读属性语义。
- Tkinter 在导入或窗口创建失败时返回 `E_TK` 并给出安装提示。

Windows 验收命令：

```powershell
py -3.12 -c "import sqlite3, socket, subprocess, tkinter; print('ok')"
py -3.12 -m unittest discover -s tests -v
py -3.12 -m edgecache.cli demo --workspace runtime\demo --port-base 46000 --gui
```
