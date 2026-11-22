# ConfigDist：多集群配置分发工作台

面向本地 Windows 环境的确定性多集群配置分发应用。它把集群、配置修订链和分发任务编译为稳定计划，在全局资源预算、集群并发配额、任务并发上限三重约束下，通过真实回环 TCP 子进程并行下发，并将权威结果保存在 SQLite 账本与内容摘要快照中。

不依赖 Docker、外部账号或线上服务。节点只绑定 `127.0.0.1` 声明端口。

## 功能覆盖

- 强校验集群、任务、端口、修订链、分片数和资源单位；首错返回 `E_SPEC`，修订环单独返回 `E_CYCLE`。
- 确定性计划编译：规划顺序、配置键和序列化均规范化，输入数组换序仍得到逐字节一致的 `plan_digest`。
- 每集群一个真实 Python 子进程；JSON 行协议；固定整数种子 + NumPy 分组聚合；回传整数毫秒耗时。
- 三重预算并行调度；无法满足约束的分片得到可判定的 `E_BUDGET`，不会永久挂起。
- 新 generation 整体替换：旧分片轨迹、摘要和预算结论统一标记为陈旧，当前视图只读取当前 generation。
- 账本逐分片单事务提交；每片记录结局、资源占用、聚合摘要和进度游标；账本带整数格式版本与内容摘要。
- 强制终止恢复只承认完整提交的分片；其余分片标记 `unknown` 并要求重跑，不伪造退出报告。
- 权威重算同时执行本地 NumPy 重算和回环节点重放，缓存不一致以 `E_PREVIEW` 定位首片。
- 快照整体替换恢复和恢复后重放；首个差异以 `E_REPLAY` 给出分片与字段；摘要错误为 `E_SNAPSHOT`。
- Tkinter 工作台提供 JSON 清单编辑、计划预览、运行、权威重算、账本、陈旧审计、快照恢复和重放。

## Windows 准备

要求 Python 3.11 或更高版本。官方 Windows Python 安装包通常已包含 Tcl/Tk；安装时保留 `tcl/tk and IDLE` 组件即可。

```bat
py -3.11 --version
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import tkinter, numpy; print('tk', tkinter.TkVersion, 'numpy', numpy.__version__)"
```

如需确认样本端口未被占用：

```powershell
Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
  Where-Object { $_.LocalPort -in 43101,43102,43103 }
```

应用本身也会在启动子进程前探测端口。活跃监听返回 `E_PORT`，节点已监听但未完成 JSON 握手返回 `E_HANDSHAKE`。

## 快速开始

```bat
python -m configdist validate --spec sample/spec.json
python -m configdist plan --spec sample/spec.json --out artifacts/sample-plan.json
python -m configdist run --spec sample/spec.json --ledger artifacts/ledger.sqlite3 --snapshot artifacts/session.snapshot.json
python -m configdist preview --ledger artifacts/ledger.sqlite3
python -m configdist verify --ledger artifacts/ledger.sqlite3
python -m configdist replay --snapshot artifacts/session.snapshot.json
python -m configdist gui --spec sample/spec.json
```

完整一站式验收：

```bat
run_demo.bat
```

或：

```bat
python -m unittest discover -v
python -m configdist demo --out artifacts/demo/acceptance.txt
```

## 清单格式

完整样本见 `sample/spec.json`。核心结构：

```json
{
  "format_version": 1,
  "simulation_seed": 20261122,
  "clusters": [
    {"id": "edge-a", "host": "127.0.0.1", "port": 43101,
     "nodes": ["a-node-1", "a-node-2"], "concurrency": 2}
  ],
  "revisions": [
    {"cluster_id": "edge-a", "version": 1, "parent": null,
     "config": {"mode": "boot", "region": "east", "policy": "strict"}}
  ],
  "tasks": [
    {"id": "deploy-core", "config_keys": ["mode", "region"],
     "target_clusters": ["edge-a"], "shards": 4,
     "resource_units": 2, "max_parallel": 3}
  ],
  "budgets": {"global_resource_units": 7, "default_task_parallel": 2},
  "versions": {"clusters": 1, "revisions": 1, "budgets": 1}
}
```

约束：

- 集群标识、任务标识分别全局唯一，且匹配 `[A-Za-z0-9][A-Za-z0-9._-]{0,63}`。
- `host` 必须字面量 `127.0.0.1`；端口必须是 `1..65535` 整数且全局唯一。
- 每个集群至少一个修订；版本号在集群内唯一并递增；父版本必须存在、更小且无环。
- 任务配置键非空且不能重复；每个目标集群的当前修订必须包含全部键。
- `shards`、`resource_units`、`concurrency`、`max_parallel`、预算均为非负整数。

## 常用命令

| 命令 | 用途 |
| --- | --- |
| `validate` | 只校验并显示计划摘要 |
| `plan` | 写出规范化计划 JSON |
| `run` | 启动节点、整体替换 generation、调度、逐片提交账本 |
| `preview` | 从账本缓存读取当前 generation 进度与摘要 |
| `verify` | 本地 NumPy 重算，并默认通过回环节点逐片重放 |
| `snapshot` / `restore` | 保存 / 识别并整体恢复快照 |
| `replay` | 恢复后重新下发已完成分片，首差报 `E_REPLAY` |
| `recover` | 强制终止后只恢复完整提交记录 |
| `gui` | 启动 Tkinter 工作台 |
| `demo` | 执行计划、三重预算、替换、重放完整验收 |

`run --crash-after-commits N` 是强制终止验收钩子：提交 `N` 个分片后调用 `os._exit(17)`。正常运行不要使用。

## 错误码

| 错误码 | 含义 |
| --- | --- |
| `E_SPEC` | 清单结构或字段违反首错规则 |
| `E_CYCLE` | 修订链父引用成环 |
| `E_PORT` | 回环端口被活跃监听占用，或节点无法绑定 |
| `E_HANDSHAKE` | 已建立连接但 JSON 握手/回传超时或失败 |
| `E_BUDGET` | 分片永久无法满足三重预算，明确标记为 blocked |
| `E_PREVIEW` | 账本缓存与本地 NumPy 权威摘要不一致 |
| `E_LEDGER` | 账本格式版本或内容摘要错误 |
| `E_RECOVERY` | 没有可恢复运行 |
| `E_REPLAY` | 恢复重放出现首个分片字段差异 |
| `E_SNAPSHOT` | 快照格式或内容摘要不一致 |
| `E_TK` | 当前 Python 无法导入 Tkinter |

## 已验证结果

本仓库已在当前 Linux/WSL 环境用真实回环子进程验证；Tkinter 代码在目标 Windows 上使用标准库，当前容器未安装 Tk，因此未执行窗口自动化。验证记录：

- 测试：`artifacts/test-results.txt`，10 项测试全部通过。
- 验收演示：`artifacts/demo/acceptance.txt`。
- 样本计划：`artifacts/sample-plan.json`。

样本关键结果：

- 18 个分片全部完成。
- 峰值：全局 `7/7` 资源单位，集群并发 `edge-a=2/2`、`edge-b=1/1`、`edge-c=2/2`，任务并发 `deploy-core=3/3`、`audit-policy=2/2`。
- 权威重算：18/18；回环节点重放 18/18；快照恢复重放 18/18。
- 修订从 `edge-a v2 -> v3` 且 `deploy-core` 移除 `edge-b` 后，旧 18 片全部陈旧，新 generation 14 片完成，双路权威重算 14/14。

## 目录

```text
configdist/        领域、校验、规划、节点、调度、账本、快照、GUI、CLI
tests/             验收测试
sample/spec.json   本地确定性样本
artifacts/         计划、测试记录、验收输出；运行数据在子目录
```

工程细节和协议见 `TECHNICAL_DESIGN.md`。
