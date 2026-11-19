# CDN 边缘缓存一致性定点诊断管线

这是一个从零构建、完全本地运行的确定性 CDN 边缘缓存模拟与验证工具。它不访问真实站点，不使用 Docker，也不依赖数据库服务。源站与至少三个边缘节点以独立子进程运行在 `127.0.0.1` 的高位端口上，通过 JSON 行协议通信；诊断账本使用 Python 标准库自带 SQLite。

核心目标是让数值口径可解释、可复现：

- 命中率、版本一致率、剩余 TTL 与派生权重统一量化为 `10^k` 网格上的定点整数。
- 所有比率乘加使用整数除法，明确采用 `floor_toward_negative_infinity`。
- 每个指标给出单值误差界、跨节点聚合误差界、分段贡献与主导误差来源。
- 多级派生链按整数系数逐级传播权重误差，并给出保守上界。
- 每批次账本、不可变快照、整体替换恢复与逐项重放校验位于同一 SQLite 文件。

## 目录

```text
cdn_edge/                 管线、源站/节点子进程、SQLite、验证与重放
cdn_edge/data/            安装后演示使用的同内容样本副本
samples/sample_spec.json  确定性内容、节点策略、请求序列与版本更新
scripts/                  演示、账本验证、环境检查与 Windows 批处理入口
tests/                    标准库 unittest 集成与异常路径测试
docs/TECHNICAL.md         协议、定点公式、事务与恢复语义
docs/VERIFICATION.md      本轮实际执行结果
```

## 环境

- Python 3.11 或更高版本。
- `sqlite3` 使用 Python 标准库，无需外部服务。
- `numpy` 是任务环境建议安装的数值依赖，但本实现的定点核心只使用 Python 大整数，安装它不是运行前提。
- `tkinter` 仅用于可选的账本查看器；Windows 官方 CPython 安装时勾选 `tcl/tk` 即可。
- 端口必须位于 `1024..65535`，并由 `samples/sample_spec.json` 或参数显式声明。

Windows 建议命令：

```bat
py -3.11 -m pip install -r requirements.txt
python scripts\check_environment.py
python -m unittest discover -s tests -v
python -m cdn_edge demo
```

WSL/Linux 下当前代码使用同一入口：

```bash
python3 -m cdn_edge demo
```

如果本机没有 `python` 命令，直接使用 Windows 的 `py -3`/`python` 或 Linux 的 `python3` 即可。

## 快速演示

默认演示会把账本写入 `artifacts/demo-ledger*.sqlite3`，若文件已存在则选择下一个序号，不删除旧账本。

```bash
python3 -m cdn_edge demo --port-base 28400
```

演示顺序执行三次会话：

1. `cold-fill`：冷启动填充，第一批请求全部回源。
2. `version-update`：内容 `base-v1 -> base-v2`，派生内容 ETag 自动变化；TTL 内旧条目返回旧版本并降低版本一致率，过期条目整体替换并递增条目版本。
3. `grid-perturb`：以 `10^2` 取代 `10^3` 精度网格，验证不可复现边界与整体替换。

演示随后自动执行：

- 独立账本自洽验证；
- 从更新会话的旧批次快照整体替换当前活动缓存/统计/误差结论；
- 从 genesis 快照重放全部批次；
- 从第一批快照重放后续批次。

本轮实测的关键值：

```text
version-update / batch 2:
  hit_rate                  = 1000 / 1000
  version_consistency_rate  = 499 / 1000
  version_consistency bound = 2 fixed units
  精确值                    = 500 / 1000
  不一致请求                = 3
```

`499` 不是精确比率本身，而是三个节点的定点比率再按请求数做跨节点整数聚合后的结果；它与精确值相差 1 个单位，落在声明上界 `2` 内，正是误差传播要显式展示的情况。

## 命令行

```bash
# 强校验内容、缓存键字段、容量、逐出规则、派生图和网格
python3 -m cdn_edge spec-check --spec samples/sample_spec.json --grid 3

# 启动源站与 3 个节点，执行可重复请求序列并逐批提交账本
python3 -m cdn_edge run \
  --spec samples/sample_spec.json \
  --ledger artifacts/run.sqlite3 \
  --grid 3

# 独立读取账本，校验批次统计、误差界、主导误差段和逐请求记录
python3 -m cdn_edge verify --ledger artifacts/run.sqlite3

# 列出会话、快照与当前恢复状态
python3 -m cdn_edge snapshots --ledger artifacts/run.sqlite3 --session sample-session

# 整体替换恢复；strict 模式在内容摘要或网格不一致时返回 E_SNAPSHOT
python3 -m cdn_edge restore \
  --ledger artifacts/run.sqlite3 \
  --snapshot-id snap-... \
  --mode replace

# 从快照恢复节点状态后重放并逐项比较，首个差异以 E_REPLAY 报明
python3 -m cdn_edge replay \
  --ledger artifacts/run.sqlite3 \
  --snapshot-id snap-...

# 可选 tkinter 查看器
python3 -m cdn_edge gui --ledger artifacts/run.sqlite3
```

Windows 批处理入口：

```bat
scripts\demo.bat --port-base 28400
scripts\validate_ledger.bat --ledger artifacts\demo-ledger.sqlite3
```

## 异常注入

```bash
python3 -m cdn_edge demo --inject cycle     # E_CYCLE
python3 -m cdn_edge demo --inject grid      # E_GRID
python3 -m cdn_edge demo --inject snapshot  # E_SNAPSHOT
python3 -m cdn_edge demo --inject replay    # E_REPLAY
python3 -m cdn_edge demo --inject port      # E_PORT
```

所有异常输出为稳定 JSON，至少包含 `ok=false`、`code`、`message`、出错 `path` 与 `field`。退出码：

| 错误码 | 退出码 | 含义 |
|---|---:|---|
| `E_SPEC` | 2 | 内容/策略/请求声明非法 |
| `E_CYCLE` | 3 | 派生关系成环 |
| `E_PORT` | 4 | 回环端口被占用或不可用 |
| `E_GRID` | 5 | 缩放因子不是合法的 `10^k` 整数幂 |
| `E_SNAPSHOT` | 6 | 快照摘要/身份与声明期望不符 |
| `E_REPLAY` | 7 | 重放或账本自洽校验出现差异 |
| `E_STALE` | 8 | 陈旧条目命中标记，不是进程级协议失败 |
| `E_ORIGIN` | 9 | 源站不可达或拒绝请求 |
| `E_PROTOCOL` | 10 | JSON 行协议或工作进程失败 |
| `E_STATE` | 11 | 账本、会话或恢复状态不一致 |

`E_CYCLE` 是 `E_SPEC` 拒绝中的专用可诊断子类，单独返回以满足异常注入与路径定位要求。

## 强校验和确定性

- 路径唯一；ETag 非空；TTL 为正整数；派生系数为正整数。
- 派生引用必须存在、同一内容内不得重复引用、整图无环；环会给出闭环路径。
- 缓存键字段限定为 `path/query/variant/tenant/accept_encoding/region`，互不重复且必须包含 `path`。
- 容量为正整数，逐出规则为 `LRU` 或 `FIFO`，节点数至少 3，源站与节点端口唯一。
- 请求 `at_ms` 非递减；源版本按 `effective_from_ms <= at_ms` 选择，结果与并发交错无关。
- 节点在返回响应前把条目、计数器和请求日志提交到自己的 SQLite 账本。强制终止后重启只从账本加载；工具不生成或伪造退出报告。
- 每批次在同一 SQLite 事务中写入批次统计、节点指标、误差界、内容摘要、游标、逐请求记录与不可变快照。

更完整的公式和事务语义见 [docs/TECHNICAL.md](docs/TECHNICAL.md)。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

测试覆盖：

- 定点下取整、单值/比较误差界、稳定决胜排序；
- `E_SPEC` 字段定位、`E_CYCLE`、`E_GRID`；
- 多级派生权重与跨节点聚合误差归因；
- 请求顺序与节点处理顺序无关性；
- 完整会话、版本更新、TTL 过期、`E_STALE`；
- 节点 `kill` 后仅凭持久化账本恢复；
- `E_PORT`；
- 快照只读触发器、内容摘要变化时整体替换、`E_SNAPSHOT`；
- 重放差异首批次/节点/字段定位 `E_REPLAY`；
- 三次演示会话与三类注入异常。
