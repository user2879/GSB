# Windows 10/11 本地运行手册

## 环境准备

```powershell
python --version          # 需 3.11+
pip install numpy
```

Tkinter 随 Python 官方安装包自带，无需额外安装。全部通信只使用 127.0.0.1
高位端口，不访问外部网络，不使用容器。

## 1. 生成确定性样本数据

```powershell
python scripts\generate_data.py --out runtime\data --revisions 8 --nodes 3 --base-port 49152
```

生成 `manifest.json`、`nodes.json`、`requests.json`。运行前确认端口未被占用：

```powershell
netstat -ano | findstr 49152
```

## 2. 运行一次完整下发会话

```powershell
python -m configlease serve --workdir runtime\run `
  --manifest runtime\data\manifest.json `
  --nodes runtime\data\nodes.json `
  --requests runtime\data\requests.json `
  --global-budget 8 --node-budget 4
```

输出每个修订的应用结果、NumPy 摘要、预算消耗与快照内容摘要。
数据库位于 `runtime\run\platform.sqlite3`，日志位于 `runtime\run\logs\`。

## 3. 异常注入（仅改输入/参数）

```powershell
# E_MANIFEST：清单损坏
python -m configlease serve --workdir runtime\inj1 `
  --manifest runtime\data\manifest.json --nodes runtime\data\nodes.json `
  --requests runtime\data\requests.json --inject-manifest

# E_PORT：端口预占用
python -m configlease serve --workdir runtime\inj2 `
  --manifest runtime\data\manifest.json.bak --nodes runtime\data\nodes.json `
  --requests runtime\data\requests.json --inject-port

# E_VERSION：快照版本冲突（先正常 serve 生成快照）
python -m configlease reopen --workdir runtime\run `
  --nodes runtime\data\nodes.json --inject-version
```

## 4. 强制终止恢复

```powershell
python -m configlease recover --workdir runtime\run --nodes runtime\data\nodes.json
```

`acceptance.py` 的 forced_termination 场景会真实 `killpg` 平台进程组，
再以新进程仅依据已提交 SQLite 记录恢复。Windows 上等价做法是任务管理器或
`Stop-Process -Id <pid> -Force` 结束平台进程树。

## 5. 重放校验与日志检索

```powershell
python -m configlease reopen --workdir runtime\run --nodes runtime\data\nodes.json
python -m configlease logs --workdir runtime\run --code E_REV
python -m configlease logs --workdir runtime\run --node node-1 --rev 1
```

## 6. 测试与端到端验收

```powershell
python -m unittest discover -s tests -v
python scripts\acceptance.py --workdir runtime\acceptance --data runtime\data
```

## 平台差异说明

- 进程终止：POSIX 用 `os.killpg` + SIGKILL；Windows 上应使用
  `subprocess` 任务对象（Job Object）或 `taskkill /PID /T /F` 结束进程树。
  节点 `NodeProcess.kill()` 使用 `proc.kill()`，在 Windows 上对应 TerminateProcess。
- 单调时钟：Python `time.monotonic_ns()` 在两端均为高分辨率单调时钟，
  同机平台与节点共享相同时间基准，TTL 毫秒语义一致。
- 路径：代码统一使用 `pathlib.Path`，SQLite 与日志目录按相对/绝对路径均可创建。
