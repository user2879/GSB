# 实际验证记录

验证日期：2026-11-22（Asia/Shanghai）。

## 1. 环境

- Python：3.11.0rc1
- NumPy：1.26.4
- SQLite：Python 标准库 `sqlite3`
- 网络：仅 `127.0.0.1`，无外部网络或账号
- 容器：未使用 Docker 或其他容器
- tkinter：当前 Linux 测试 Python 未安装；`gui` 命令已实测返回 `E_ENV`。Windows 官方 Python 安装包应包含 tkinter。

## 2. 自动化测试

命令：

```powershell
python -m unittest discover -s tests -t . -v
```

实际结果：

```text
Ran 8 tests in 4.206s
OK
```

覆盖：

1. 规格合法加载及 `E_REF`、`E_TIMING` 注入。
2. 重复端口返回首个 `E_SPEC` 字段 `clusters[1].port`。
3. 优先级接管、续约、过期、释放、确定性迁移序列。
4. 服务端单调时钟真实到期。
5. 真实 TCP 套接字握手超时 `E_HANDSHAKE`。
6. 握手回执缺字段 `E_REPLY`。
7. 回环端口占用 `E_PORT`。
8. 端到端确定性、参数演进、快照重放、`E_REPLAY`、强制终止恢复和续跑一致性。

## 3. 端到端验收

命令：

```powershell
python scripts/run_acceptance.py
```

13 项全部通过：

| 检查 | 结果 |
| --- | --- |
| 规格加载 | PASS，14 事件、2 集群 |
| 确定性唯一裁决 | PASS，两次全新数据库结果一致 |
| 持有权起止区间与容量不变量 | PASS，开放区间数与有效租约数一致 |
| 到期回收与容量约束 | PASS |
| 已提交事件幂等重跑 | PASS，处理事件数 0 |
| 容量/时长/优先级演进整体替换 | PASS，两集群版本 1 -> 2 |
| 快照重放一致性 | PASS |
| 快照差异整体恢复并报 `E_SNAPSHOT` | PASS |
| 篡改重放报 `E_REPLAY` 并定位首差异 | PASS，`alpha` / 事件 1 / `decision` |
| `E_REF`、`E_TIMING`、`E_SNAPSHOT` 注入 | PASS |
| 结构化日志保留业务码 | PASS |
| 强制终止恢复与未提交区间重跑 | PASS，未知区间 5..8，续跑 5..14 |
| 回环端口占用 | PASS，`E_PORT` |

关键摘要：

```text
事件流摘要:
9146963274e26298f915e3b952fe239a4719f4e0a8ee468901c9aaab851e905d

两次全新运行的裁决回执序列摘要:
82dc9afc080900e26ab798867b29a48d2e81002b792a81ee5b0725f04fe97db0

初始账本内容摘要:
2ca12e136d69a92f7430447879af4f45e6402ae2892013c886be565e4a5616eb

演进后账本内容摘要:
24da09e8d3fd8fbdabbaa37039dfedf6753692aefc4089f4b0c0f6d25eeb7390

演进后快照内容摘要:
d538786adacee99a3263e91b821cd8c62db5ed423571de404611eccadb683558
```

完整机器可读结果：`artifacts/acceptance-summary.json`。每次运行的数据库、报告、日志和快照位于 `artifacts/acceptance/<UTC时间戳>/`。

## 4. 异常码实测样例

`E_REF`：

```json
{"ok":false,"code":"E_REF","field":"holders[2].targets[2]","message":"target cluster does not exist: __injected_missing_cluster__"}
```

`E_TIMING`：

```json
{"ok":false,"code":"E_TIMING","field":"clusters[0].lease_duration_ms","message":"lease_duration_ms must be greater than renewal_interval_ms"}
```

`E_SNAPSHOT` 摘要注入：

```json
{"ok":false,"code":"E_SNAPSHOT","field":"content_digest","message":"snapshot content digest does not match its contents"}
```

`E_REPLAY`：

```json
{"ok":false,"code":"E_REPLAY","field":"decision","details":{"cluster_id":"alpha","event_seq":1,"expected":"rejected","actual":"granted"}}
```

`E_PORT`：

```json
{"ok":false,"code":"E_PORT","field":"port","details":{"cluster_id":"alpha","port":39171}}
```

## 5. 真实时钟过期

独立服务测试使用 250 ms 租约、100 ms 续约间隔、`hybrid` 时钟。授予后等待 450 ms，再发送 `status`，结果：

```text
status_used 0
reason lease_expired
REAL_CLOCK_EXPIRY_OK
```

这验证了服务端 `time.monotonic_ns()` 真实计时路径，而确定性样本仍使用事件时钟重放。

## 6. 确定性复核

同一 14 事件序列分别以 `batch-size=4` 和 `batch-size=3` 在全新数据库中运行。两次结果：

- `decision_trace` 完全相同。
- `reply_sequence_digest` 完全相同。
- `ledger_content_digest` 完全相同。

默认 `hybrid` 模式也在 5000 ms 样本租约下做了两次快速复核，两个摘要一致；正式验收使用 `logical` 模式以消除壁钟调度对确定性断言的影响。

## 7. GUI 环境检查

当前容器执行：

```powershell
python -m cluster_lease gui --db build/state-run1.db
```

返回：

```json
{"ok":false,"code":"E_ENV","field":"tkinter","message":"tkinter is not available in this Python installation: No module named 'tkinter'"}
```

该结果证明核心 CLI 在无 GUI 环境下可诊断退出。Windows 官方 Python 应使用同一条命令启动只读诊断界面。
