# 运行手册（Windows 本地）

## 一次性准备

```powershell
py -3 --version          # 需 3.11+
py -3 -m venv .venv
.venv\Scripts\python scripts\generate_samples.py
```

## 演示与验收

```powershell
# 完整业务演示（归一/降级/补齐、TTL、LRU、TCP 转发、整体替换、快照）
.venv\Scripts\python scripts\demo.py

# 三类异常注入，分别得到 E_ZONE / E_VER / E_TXN
.venv\Scripts\python scripts\demo.py --inject zone-corrupt
.venv\Scripts\python scripts\demo.py --inject cache-version
.venv\Scripts\python scripts\demo.py --inject upstream-txn

# 单元测试
.venv\Scripts\python -m unittest discover -s tests -v
```

## 联调上游模拟（独立终端）

```powershell
# 默认 u2（EDNS + DO）；--u1 退化为 u1；--bad-txn / --bad-chain 注入异常
.venv\Scripts\python scripts\mock_upstream.py --port 5354
```

## 工作台

```powershell
.venv\Scripts\python scripts\run_workbench.py                 # GUI + 回环服务
.venv\Scripts\python scripts\run_workbench.py --serve         # 无界面服务
```

回环端口默认 UDP/TCP 5353、上游 TCP 5354，均可通过命令行参数改为其他高位端口。
所有数据来自 `dns_workbench/samples/` 本地确定性样本，不访问真实域名或线上解析器。

## 产物

- `data/run/workbench.jsonl`：结构化 JSON Lines 日志（含错误码、字段、偏移、上下文）。
- `data/run/ledger.sqlite3`：快照库（`meta`、`entries`、`hits` 三表）。
- `dns_workbench/samples/`：区文件、三版本缓存、u1/u2 应答、线格式报文样本。
