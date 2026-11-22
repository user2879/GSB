# CDN 边缘缓存一致性诊断台

面向本地 Windows 环境（同时兼容 Linux/macOS）的 CDN 边缘缓存一致性诊断台。
在 `127.0.0.1` 高位端口上以子进程模拟一个确定性内容源与三个边缘缓存节点，
节点通过 JSON 行协议向源站拉取内容并回报缓存条目状态；缓存条目、分片下发
租约与传播轨迹以本地 SQLite 持久化为账本，支持快照保存、版本识别与整体
替换恢复，并提供恢复后的一致性重放校验与三类异常注入。

## 环境要求

- Python 3.10+（Windows 安装版自带 `tkinter` 与 `sqlite3`）
- `pip install -r requirements.txt`（NumPy，用于诊断报告统计）
- 不使用 Docker / 容器，不访问外部网络与真实账号

## 目录结构

```
cdn_diag/
├── run_demo.py            # 演示入口（三场景顺序执行 / 异常注入）
├── run_gui.py             # tkinter 诊断台入口
├── requirements.txt
├── samples/               # 确定性样本（只读声明）
│   ├── content_spec.json  #   内容声明：路径/ETag/字节内容/TTL/派生关系
│   ├── policy.json        #   缓存键字段、节点容量与逐出规则、并发配额、端口范围
│   └── requests.json      #   请求序列（批次 + 逻辑单调时钟 tick_ms）
├── src/cdn_diag/          # 源码
│   ├── spec.py            #   声明加载与强校验（E_SPEC，整体拒绝）
│   ├── derive.py          #   派生图：稳定拓扑序、下游推导、确定性组合
│   ├── cache.py           #   节点缓存：TTL 递减、LRU/FIFO 确定性逐出
│   ├── lease.py           #   租约签发/续约/过期回收/冲突裁决（E_LEASE）
│   ├── scheduler.py       #   全局+每节点并发配额、排队计时、超时重试
│   ├── origin_server.py   #   内容源子进程（JSON 行协议）
│   ├── edge_node.py       #   边缘节点子进程（崩溃后仅依账本恢复）
│   ├── orchestrator.py    #   编排：端口/握手/请求流/传播/快照/重放
│   ├── ledger.py          #   SQLite 账本：格式版本、摘要、快照、整体替换恢复
│   ├── replay.py          #   一致性重放校验（E_REPLAY 定位首个不一致）
│   ├── inject.py          #   异常注入：成环/截断/摘要篡改
│   ├── report.py          #   稳定顺序诊断报告（NumPy 统计）
│   ├── demo.py            #   三场景演示流程
│   └── gui.py             #   tkinter 交互诊断台
├── ledger/                # 运行期生成：ledger.db / snapshot.db / working_spec.json
├── logs/                  # 运行期生成：结构化 JSONL 日志与子进程 stderr
├── tests/test_acceptance.py
└── docs/                  # 实际验证输出
```

## 快速开始

```bat
:: 演示：冷启动填充 -> 版本演进派生失效 -> 快照保存与恢复重放
python run_demo.py

:: 异常注入（退出码即错误码序号）
python run_demo.py --inject cycle      :: E_GRAPH    (15)
python run_demo.py --inject truncate   :: E_SNAPSHOT (16)
python run_demo.py --inject digest     :: E_VERSION  (17)

:: 交互诊断台（需 tkinter）
python run_gui.py

:: 验收测试
python -m unittest discover -s tests -v
```

演示前自动清理 `ledger/` 与 `logs/`，冷启动从空状态开始；端口在
`policy.json` 的 `port_range`（默认 49152-49252）内动态探测分配，避免固定冲突。

## 关键设计

- **确定性**：缓存 TTL 以请求序列携带的逻辑单调时钟 `tick_ms` 递减，相同
  请求序列与容量策略在多次运行中得到逐字节一致的命中与逐出结论；租约过期
  与调度排队计时使用真实 `time.monotonic`。
- **两阶段批次**：每批请求先集中查找、再统一经调度器并发回源（全局
  `global_inflight` + 每节点 `per_node_inflight` 配额，耗尽排队并记录等待
  起止毫秒）；重放按相同阶段语义驱动。
- **派生失效传播**：ETag 变化沿派生图推导全部直接/间接下游，按稳定拓扑序
  （Kahn + 字典序候选）逐节点整体替换，同一路径一次传播仅替换一次，轨迹
  记录触发源、受影响路径与替换前后条目版本号。
- **租约冲突裁决**：同一路径同一节点出现指向不同内容版本的有效租约时，
  显式稳定规则 `newer_content_version_wins` 撤销旧租约并记录依据；续约凭
  租约版本号乐观校验，不符以 `E_LEASE` 拒绝。
- **账本快照**：快照携带整数格式版本与内容摘要；恢复时校验完整性
  （截断/损坏 -> `E_SNAPSHOT`）、摘要一致性（不符 -> `E_VERSION`），并比较
  快照内容集合与磁盘当前声明，不一致则整体替换全部条目、租约与派生关系
  （不做增量合并），旧快照保留为只读，稳定列出新增/删除/版本变更路径。
- **重放校验**：恢复后按账本记录的请求序列与内容版本重新驱动节点，逐请求
  比较命中来源、返回内容版本、剩余 TTL、逐出事件与传播轨迹，首个不一致
  定位到批次/节点/路径/字段并以 `E_REPLAY` 报明依据。

## 错误码

| 代码 | 含义 | 退出码 |
|---|---|---|
| E_SPEC | 声明校验失败（首个违规路径与字段） | 10 |
| E_PORT | 回环高位端口被占用 / 无可用端口 | 11 |
| E_HANDSHAKE | 握手真实套接字超时 | 12 |
| E_PROTO | 握手回执缺字段 / 协议错误 | 13 |
| E_LEASE | 租约版本不符 / 过期 / 路径越界 | 14 |
| E_GRAPH | 运行期派生关系成环 | 15 |
| E_SNAPSHOT | 快照损坏 / 截断 / 格式版本不兼容 | 16 |
| E_VERSION | 内容版本摘要不符 | 17 |
| E_REPLAY | 重放与账本不一致 | 18 |

## 实际验证结果（本机 Linux，Python 3.11）

- `python run_demo.py`：三场景顺序执行成功，退出码 0；重放校验
  `verdict=OK compared_fields=112`；连续两次运行命中与逐出结论逐字节一致。
  完整输出见 `docs/verification_demo_output.txt`。
- `python run_demo.py --inject cycle`：输出 `E_GRAPH`（环路径
  `/index.html -> /page/home -> /page/app`），退出码 15。
- `python run_demo.py --inject truncate`：输出 `E_SNAPSHOT`
  （`database disk image is malformed`），退出码 16。
- `python run_demo.py --inject digest`：输出 `E_VERSION`
  （记录摘要 `tampered-digest` 与计算摘要不符），退出码 17。
  注入输出见 `docs/verification_inject_*.txt`。
- `python -m unittest discover -s tests`：21 个用例全部通过，覆盖
  E_SPEC/E_PORT/E_HANDSHAKE/E_PROTO/E_LEASE/E_GRAPH/E_SNAPSHOT/E_VERSION/
  E_REPLAY、逐出确定性、TTL 递减与过期、租约冲突裁决、版本漂移整体替换。
- tkinter GUI（`run_gui.py`）已通过编译校验；本验证环境无系统 Tk 库，
  交互界面请在 Windows 本地运行（Windows 版 Python 自带 tkinter）。
