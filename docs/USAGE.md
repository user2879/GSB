# CloudShard 本地演示与独立校验

这套代码交付一个可编辑安装的 Python 包 `cloudshard`。它不访问网络数据集、容器、真实账号或付费服务；所有节点都是本机 `127.0.0.1` 子进程，样本由声明中的整数种子和稳定分片序号确定性生成。

## 1. 安装

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
```

如果镜像裁剪了 `ensurepip`（本机验收时遇到），可用已安装的 `virtualenv`：

```bash
virtualenv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
```

Ubuntu/WSL 的 Tkinter 系统依赖为：

```bash
sudo apt install python3-tk tk8.6
```

若系统 `python3-tk` 面向发行版默认 Python 3.10，而解释器是另一套 Python 3.11，需要安装匹配 Python 3.11 的 Tk 包。当前工作机无 sudo 密码，验收时另用 `apt-get download` 将 `libtk8.6`、`libtcl8.6`、`libxft2` 等解包到 `artifacts/tk-root/`；`scripts/tk-view.sh` 会自动设置 `LD_LIBRARY_PATH`、`TCL_LIBRARY` 和 `TK_LIBRARY`。

## 2. 一条命令完成验收

```bash
./scripts/demo.sh
```

该脚本依次执行：

- 安装包和测试依赖；
- 生成空闲回环端口声明并运行 3 个真实节点子进程；
- 完成首版 8 个分片和修订版 2 的 5 个失效分片；
- 运行独立账本校验；
- 重放事件并检查任意分批边界的归属字节一致；
- 运行包含真实 `SIGKILL` 恢复、`E_PORT`、`E_TIMEOUT`、`E_FIELD`、`E_LEASE` 的验收；
- 运行全部 pytest。

也可以分步：

```bash
cloudshard demo --workdir artifacts/demo
cloudshard verify \
  --spec artifacts/demo/declaration.generated.json \
  --ledger artifacts/demo/ledger.sqlite
cloudshard acceptance --workdir artifacts/acceptance
pytest -q
```

## 3. 使用固定样本

`samples/declaration.json` 使用固定高位端口 `39111-39113`。如果端口被占用，可重新生成空闲端口样本：

```bash
cloudshard init-sample --out samples/declaration.json
cloudshard validate --spec samples/declaration.json
cloudshard run --spec samples/declaration.json --workdir artifacts/sample-run
cloudshard verify --spec samples/declaration.json \
  --ledger artifacts/sample-run/ledger.sqlite
```

## 4. 独立校验命令

```bash
cloudshard verify --spec <原始声明.json> --ledger <账本.sqlite>
```

校验命令以 SQLite 只读 URI 打开账本，不读取快照、不连接节点、不依赖运行中进程。它会：

- 校验声明、依赖 DAG、修订链和事件序号；
- 审计整数 `format_version=1` 与内容摘要；
- 从原始声明和节点种子重算样本及 NumPy 聚合摘要；
- 重推声明事件和运行时事件中的授予、续约、释放、到期、配额回收和崩溃恢复；
- 比对归属序列、失效集合、未受影响分片的逐字节结论、进度游标和恢复后的重跑记录。

失败时输出 `E_VERIFY`，并尽量给出首个不一致的 `shard_id`、`field` 与 `basis`。

## 5. 崩溃恢复演练

一键验收会启动如下真实子进程：

```bash
cloudshard run \
  --spec artifacts/acceptance/crash/spec.json \
  --workdir artifacts/acceptance/crash \
  --crash-before-commit s2 \
  --crash-marker artifacts/acceptance/crash/crash-marker.json
```

父验收进程看到 marker 后对整个新进程组发送 `SIGKILL`。此时一个结局事务已经 `BEGIN` 且插入未提交行，但尚未 `COMMIT`；SQLite 在进程死亡后回滚该事务。此前已提交的 `attempts/occupancy/leases` 仍显示 `DISPATCHED/HELD/active`。重新执行任意流水线或：

```bash
cloudshard recover --ledger <账本.sqlite>
cloudshard run --spec <声明.json> --workdir <目录>
cloudshard verify --spec <声明.json> --ledger <账本.sqlite>
```

恢复器先把这些分派标记为 `UNKNOWN_RERUN`、回收占用，并在同一 SQLite 事务写入可重放的 `lease_release/reclaimed/crash_recovery` 事件和归属记录，然后才允许调度器重跑。系统不会、也不应声称能生成被 `SIGKILL` 瞬间的退出报告。

## 6. Tkinter 查看器

```bash
./scripts/tk-view.sh artifacts/demo/ledger.sqlite artifacts/demo/snapshot.json
```

查看器只以只读模式查询 `current_view`。无图形显示时 Tk 根窗口可能无法启动，但 `import tkinter` 和 Tcl/Tk 版本检查可在无头环境完成。

## 7. 错误码

- `E_SPEC`：JSON/schema/引用/DAG/修订链不合法，包含首个 `path` 与 `field`。
- `E_PORT`：回环端口无法绑定或无进程响应。
- `E_TIMEOUT`：真实 TCP 连接、握手或 JSON-line 读取超过套接字超时。
- `E_FIELD`：握手或回执缺少/错配字段。
- `E_LEASE`：租约版本不符、续约超上限、租约失效或容量权益冲突。
- `E_VERIFY`：独立重放与账本不一致。
- `E_ENV`：可选环境能力（当前用于 Tkinter）不可用。
