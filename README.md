# Shardflow

面向计算引擎多节点分片任务的本地确定性编排包：独立子进程节点、回环 TCP
JSON 行协议、整数定点 NumPy 聚合、可续约/可过期租约、三重资源约束调度、
SQLite 持久化账本、强制终止恢复、独立重放校验与 Tkinter 诊断界面。

## 环境

- Windows 官方 Python 3.11 或 3.12（开发与测试也兼容 POSIX Python 3.11）。
- 依赖：`numpy==1.26.4`。Tkinter 随 Windows 官方 Python 安装包提供。
- 仅使用 `127.0.0.1` 回环端口和本地文件，不访问外部站点。

```powershell
python -m pip install -r requirements.txt
```

## 快速开始

```powershell
python examples/generate_demo.py
python -m shardflow.cli validate demo/demo.spec.json --events demo/demo.events.ndjson
python -m shardflow.cli run demo/demo.spec.json --events demo/demo.events.ndjson --ledger demo/demo.ledger.sqlite
python -m shardflow.cli verify demo/demo.spec.json --events demo/demo.events.ndjson --ledger demo/demo.ledger.sqlite
python -m shardflow.cli inspect --ledger demo/demo.ledger.sqlite
python -m shardflow.cli diagnose demo/demo.spec.json --events demo/demo.events.ndjson --ledger demo/demo.ledger.sqlite
```

可重复演示（完整运行、节点强制终止恢复、账本篡改整体替换）：

```powershell
python examples/demo_run.py all
```

## 目录

- `shardflow/`：可导入 Python 包与 CLI。
- `examples/generate_demo.py`：生成 demo JSON 声明与 NDJSON 事件。
- `examples/demo_run.py`：三模式可重复演示。
- `tests/`：32 个单元、协议、账本与真实子进程集成测试。
- `demo/`：生成的声明、事件、账本与演示日志。
- `docs/ARCHITECTURE.md`：架构、协议、恢复语义与错误码说明。

## 错误码

| 代码 | 含义 |
| --- | --- |
| `E_SPEC` | 声明或事件流违规，含首个对象与字段 |
| `E_PORT` | 回环端口占用或无法连接 |
| `E_HANDSHAKE` | 真实套接字超时内握手未完成 |
| `E_FIELD` | 线协议回执缺字段或字段非法 |
| `E_LEASE` | 乐观租约版本不符或续约上限耗尽 |
| `E_DUP` | 重复取消或对终态分片取消 |
| `E_LEDGER` | 账本格式版本/内容摘要不一致（严格模式） |
| `E_REPLAY` | 独立重放与账本首个不一致点 |

## 测试

```powershell
python -m unittest discover -s tests
```

更多设计与验证细节见 `docs/ARCHITECTURE.md`。
