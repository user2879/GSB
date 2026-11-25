# LeaseMesh 使用说明

首版实现位于 `leasemesh/`，确定性样本位于 `samples/`，自动化验收位于 `tests/`。

## 环境

```bash
python3 --version
python3 -m pip install -r requirements.txt
```

SQLite 使用 Python 标准库 `sqlite3`。Tkinter 为可选面板；当前 WSL 若提示 `libtk8.6.so` 缺失，执行：

```bash
sudo apt-get update
sudo apt-get install -y python3-tk
```

无头 CLI、TCP 节点、SQLite 账本和测试不依赖 Tkinter。

## 常用命令

```bash
python3 -m leasemesh validate \
  --declarations samples/declarations.json \
  --requests samples/requests.json \
  --events samples/events.json

python3 -m leasemesh check-ports --declarations samples/declarations.json

python3 -m leasemesh adjudicate \
  --declarations samples/declarations.json \
  --events samples/events.json \
  --out work/adjudication.json

python3 -m leasemesh evolve \
  --old samples/declarations.json \
  --new samples/declarations.v2.json \
  --events samples/events.json \
  --out work/evolution.json

python3 -m leasemesh execute \
  --declarations samples/declarations.json \
  --requests samples/requests.json \
  --allocate-ports \
  --workdir work/demo \
  --ledger work/demo/ledger.sqlite3 \
  --snapshot work/demo/snapshot.json \
  --reset

python3 -m leasemesh verify \
  --declarations work/demo/allocated-declarations.json \
  --ledger work/demo/ledger.sqlite3

python3 -m leasemesh recover \
  --declarations work/demo/allocated-declarations.json \
  --ledger work/demo/ledger.sqlite3 \
  --snapshot work/demo/snapshot.json \
  --rerun-manifest work/demo/rerun.json

python3 -m leasemesh inject-spec --kind duplicate_port
python3 -m leasemesh inject-stale --allocate-ports --workdir work/stale
python3 -m leasemesh tk
```

## 关键设计

- `spec.py` / `jsonio.py`：JSON 行号定位与 `E_SPEC` 强校验。
- `node_server.py`：节点子进程、JSON-line TCP、代数匹配、NumPy 定点分组校验和。
- `protocol.py`：真实套接字连接、hello 握手、读写超时和 `E_FIELD` 回执校验。
- `scheduler.py`：线程池、全局与节点级条件变量上限、单调时钟排队指标。
- `state_machine.py`：`granted -> renewed -> released|expired` 显式许可表。
- `ownership.py`：槽位唯一归属、两类回收同一代数只裁决一次、按 `seq` 规范化。
- `evolution.py`：节点/租约/代数/事件引用追踪，稳定失效集与未受影响记录逐字节证明。
- `store.py`：SQLite 哈希链、整数 `PRAGMA user_version` / `ledger_format`、预写请求与配额预留、同事务投影。
- `snapshot.py`：快照摘要校验、不一致时整体替换、只读备份和差异列表。
- `verify.py`：以只读 SQLite 连接重放全部提交并定位首个 `E_VERIFY` 差异。

强制终止时不生成也不承诺退出报告。恢复命令只依据已完整提交的 SQLite 事务恢复代数与归属；`planned` 请求被标记为 `unknown/rerun_required`，其容量预留先置为 `reclaimed`。
