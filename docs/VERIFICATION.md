# 实际验证记录

验证日期：2026-11-24

执行环境：Linux/WSL2，Python `3.11.0rc1`，NumPy `1.26.4`。当前容器没有 Windows 实机，也没有可用的 `libtk8.6.so`；Windows 子进程/套接字分支和 Tk 操作台需要在最终 Windows 目标机按 README 再跑一次。核心链路、真实回环子进程、SQLite 和 CLI 均已在本环境实跑。

## 自动化测试

命令：

```text
python -m py_compile gateway_diag/*.py tests/*.py
python -m unittest discover -s tests -v
```

实际结果：

```text
Ran 39 tests in 31.144s

OK
```

同时使用 PEP 517 无隔离模式构建 wheel，确认包内容与入口点：

```text
python -m pip wheel --no-build-isolation --no-deps -w <临时目录> .
gateway_diag-0.1.0-py3-none-any.whl
package_files=13
[console_scripts]
gateway-diag = gateway_diag.cli:main
```

测试覆盖摘要：

- `test_validation.py`：首个非法条目/字段、ID 与端口唯一、回环字面量、正整数/整数、量化集合、权重边界、目标后端、请求维度与重复序号。
- `test_routing_parameters.py`：参数同进程/跨进程逐位一致、维度/种子/量化变化来源、分桶范围与稳定、0/10000 权重边界、输入顺序无关。
- `test_backends.py`：真实子进程握手与推理、同端口重启、`E_PORT`、`E_HANDSHAKE`、`E_RECEIPT`。
- `test_scheduler.py`：全局 2/后端 1 的真实 `gateway-batch_*` 线程池上限、排队等待单调时钟、完成回调。
- `test_engine_integration.py`：真实回环端的完整运行、快速视图/权威规范字节一致、`E_PREVIEW` 去向/批归属/输出摘要/逐项与聚合耗时定位、`E_REPLAY` 请求/后端/输出字段定位、revision 整体替换与 stale 隔离。
- `test_recovery.py`：独立 CLI 进程在已提交至少一个批次后被 `SIGKILL`；重启将缺失批次标为 `UNKNOWN`；重跑后 `RESOLVED`、账本完整、重放一致，并再次通过权威校验。
- `test_cli.py`：JSON 错误输出、退出码、`run`/`preview`/`verify` 实际链路、占用端口 `E_PORT`。

## 固定样例端到端实跑

命令使用仓库内 `examples/sample_spec.json` 与 `examples/sample_requests.ndjson`，数据库放在临时目录；依次执行 `run`、`preview`、`verify --mode both`、`recover`。

实际摘要：

```text
run: ok=True revision=1 total_batches=5 newly_committed=5 global_limit=2
scheduler: max_global_observed=2
           max_backend_observed={"canary-b":1,"stable-a":1}
           worker_threads=["gateway-batch_0","gateway-batch_1"]
preview: entries=8 stable=5 canary=3 batch_count=5
verify.authority: consistent=True replayed_batches=5 replayed_requests=8
verify.replay: consistent=True ledger_integrity_ok=True
recover: initial_needs_rerun=False newly_committed=0
         final_needs_rerun=False replay_ok=True
```

本次单次快速视图规范 JSON 的摘要：

```text
SHA-256(canonical preview view)
= 4f27fb04ce108481c32c3136db9cfa47b756ac2fc2c2fe99078209c64a57c829
= verify.authority.preview_canonical_digest
```

该摘要由快速视图的 8 条请求与聚合统计重新规范序列化后计算，和权威重算输出的规范字节摘要相同；权威结果覆盖字节数为 5122。该次运行的批次等待总耗时为 65ms、执行总耗时为 40ms。

## 异常码实跑/测试映射

| 错误码 | 验证方式 | 结果 |
|---|---|---|
| `E_SPEC` | CLI 重复后端 ID、规则版本未递增、请求维度错误 | 退出码 2，定位 entry/field |
| `E_PORT` | 真实占用回环端口后启动 worker/CLI | 退出码 3，定位后端与端口 |
| `E_HANDSHAKE` | 子进程不输出 ready 行并超时 | 定位后端与端口 |
| `E_RECEIPT` | 回执缺 `output_digest` | 定位后端、端口与字段 |
| `E_PREVIEW` | 篡改缓存去向/批归属/输出摘要/批次耗时/聚合耗时 | 定位首个请求与 `target`/`batch_id`/`output_digest`/`wait_ms`/`stats.total_exec_ms` |
| `E_REPLAY` | 篡改账本输出摘要后真实重放 | 定位请求、后端与 `output_digest` |

## 强制终止恢复细节

恢复测试使用 4 个批大小为 1 的请求、全局上限 1、每后端配额 1，并让 worker 每批延迟 900ms。测试轮询 SQLite，在 `1 <= committed_batches < 4` 时对独立 CLI 进程执行 `SIGKILL`。随后：

- 打开 WAL 账本仍只看到完整提交批次。
- `recovery_report` 报告 `needs_rerun=true`，缺失批次进入 `UNKNOWN`。
- `recover` 只重跑缺失批次，提交事务将对应行改为 `RESOLVED`。
- 最终 `needs_rerun=false`，`ledger_integrity_ok=true`，`verify_replay` 与权威校验均通过。

## 未在当前容器完成的部分

- Windows 真实进程创建标志、`SO_EXCLUSIVEADDRUSE` 行为与防火墙首次提示。
- Tk 操作台窗口渲染；当前 Linux Python 缺少 `libtk8.6.so`，但 `gui.py` 已通过语法编译。Windows python.org 安装包通常自带 Tk。
- 当前基础镜像缺少 `ensurepip/python3.11-venv`，因此未在本容器创建 `.venv`；已用 PEP 517 wheel 构建验证包内容与控制台入口，Windows 目标机按 README 创建独立虚拟环境。
- 固定样例端口若被其他程序占用，按设计返回 `E_PORT`；自动化测试使用动态空闲端口，不依赖样例固定端口。
