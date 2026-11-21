# configmesh — 本地跨组件配置分发与验证平台

面向 Windows 本地环境（同样可在 Linux/macOS 运行）的资源编排数据处理管线。平台将
模拟组件节点以独立子进程启动，节点在 `127.0.0.1` 高位端口上通过 **TCP JSON 行协议**
接收配置切片，使用 NumPy 对确定性样本做整数定点聚合，并回传派生状态、聚合摘要与
整数毫秒耗时。平台负责清单强校验、定点量化与误差传播、增量失效推导、预算内并行调度、
权威重算收敛、SQLite 快照整体替换/重放以及异常注入与结构化日志。

## 环境准备

- Python 3.10+（验证环境 Python 3.11）
- `pip install -r requirements.txt`（仅依赖 NumPy；`sqlite3` 为标准库）
- Windows 官方 Python 安装包默认包含 tkinter；无 tkinter 的无头环境自动降级为
  文本预览，核心功能不受影响
- 节点仅绑定 `127.0.0.1`，默认端口段 18001–18003，可在注册表中修改

## 目录结构

```
configmesh/
  errors.py         # 9 个稳定业务码与 BusinessError
  models.py         # 节点/策略/配置项/编译产物/回执数据模型
  util.py           # 规范化 JSON、SHA256 摘要、单调时钟、稳定拓扑排序
  fixedpoint.py     # 10**k 定点量化、整数 MAC、饱和规则、误差界与稳定决胜
  sample.py         # 确定性样本与 NumPy int64 定点聚合
  manifest.py       # 注册表/版本清单强校验（全有或全无）
  compiler.py       # 逐键编译、逐字节产物、增量失效传播
  protocol.py       # JSON 行帧与节点子进程入口（python -m configmesh.protocol）
  node_process.py   # 平台侧子进程生命周期、握手、下发、E_PORT/E_HANDSHAKE/E_REPLY
  scheduler.py      # 全局+每节点信号量预算调度、排队计时、峰值并发、重试
  convergence.py    # 权威重算、逐字节收敛判定、E_PREVIEW 守卫
  preview.py        # 无头预览引擎与可选 Tkinter 窗口
  storage.py        # SQLite 快照、内容摘要、整体替换、账本
  injections.py     # 仅改输入/参数的三类异常注入
  session.py        # 状态机编排与重放校验
  cli.py            # 命令行入口
  demo.py           # 一键验收演示
examples/           # 节点注册表与三版配置清单
tests/              # 26 个 unittest 用例（含真实子进程集成测试）
```

## 快速开始

```bash
# 仅做强校验
python -m configmesh.cli validate

# 完整管线：启动节点 -> 编译 -> 并行下发 -> 收敛 -> 快照 -> 重放
python -m configmesh.cli run --workdir workdir/demo --grid-exp 3 \
    --global-limit 3 --per-node-quota 2

# 一键验收（峰值并发、收敛、快照替换、重试、三类异常码、注入后合法输入恢复）
python -m configmesh.demo

# 三类异常注入（注入只改输入文件/参数，不改源码；结束后自动还原清单）
python -m configmesh.cli inject --kind corrupt-manifest
python -m configmesh.cli inject --kind cycle
python -m configmesh.cli run --grid-exp -3        # E_GRID

# 派生项误差感知稳定决胜
python -m configmesh.cli compare --key-a heat_index --key-b flow_score

# 界面缓存漂移检测（E_PREVIEW）
python -m configmesh.cli preview-drift

# 测试
python -m unittest discover -s tests -v
```

## 定点口径与误差传播

- 会话参数 `grid_exp = k`，缩放因子 `S = 10**k`，合法范围 `k ∈ [0,9]`，越界为
  `E_GRID`。
- 量化采用**向零截断**：`Q(x) = trunc(x*S)`，结果按有符号 64 位**饱和钳制**
  （不回绕）。
- 派生项 `y = Σ c_i*x_i` 全部以整数乘加完成，再按 `q(y)=trunc(ΣQ(c_i)Q(x_i)/S)`
  重缩放；乘法与累加均做 int64 饱和。
- 误差上界（q 单位，保守整数口径）：
  `B_q = 1 + Σ_i (|Q(c_i)| + |Q(x_i)| + 1)`，其中 `1` 为重缩放截断贡献，
  每个输入项另给出 `contribution_pct` 贡献占比；真实值上界为 `B_q/S`。
- 两个派生项比较：仅当 `|gap| > bound_a + bound_b + 1` 时判定 GT/LT，否则为
  INDETERMINATE，并以字典序较小键作为确定性决胜（见 `fixedpoint.compare_derived`）。

## 增量编译与失效推导

- 编译产物按键生成，内容不含版本号；未变化键在跨版本时 SHA256 摘要逐字节一致，
  报告字段 `byte_identical_unchanged` 可逐项核对。
- 变化指纹覆盖：键值、派生系数、目标节点集合、分发策略。
- 失效集合沿派生依赖图传播直接与间接依赖，按 Kahn 稳定拓扑（字典序破平）输出，
  每步带 `propagated_from` 传播依据；新增/删除键、节点目标变化与策略变化分别标记。

## 调度与收敛

- 全局信号量（`--global-limit`）与每节点信号量（`--per-node-quota`）同时获取后
  任务才能启动，预算不可绕过；账本记录每任务入队/开始/结束单调时钟毫秒、排队时长、
  观测峰值与每次 attempt。
- 超时按节点声明的 `fixed` 策略重试；首次与重试 attempt 独立记录，单次下发消息
  语义不变（重试自动撤销超时注入标志）。
- 收敛始终以平台权威重算为准，对派生状态与聚合摘要做规范化 JSON 逐字节比对；
  缺回执或不一致为 `E_CONVERGE`，稳定列出节点、分片、期望与实际。
- UI 预览缓存仅用于展示；`E_PREVIEW` 守卫定位首个差异节点与字段，缓存不能替代
  权威结论。

## 快照与重放

SQLite 文件 `<workdir>/ledger.sqlite`：

- `snapshots` 表持久化清单版本、精度口径、状态机状态、分发计划、回执、任务账本、
  定点误差界及整包 `content_digest`（SHA256）。
- 重新打开先比较版本 + 精度 + 摘要；不一致时 `replace_all_session_state`
  **整体替换**而非增量合并。
- 重放按快照版本与精度重新编译/重算，逐项比较 `qvalue/is_derived/derived_from_q/
  error_bound/artifact_digest`、回执集合与收敛结论；首个不一致键与字段以
  `E_REPLAY` 定位。

## 业务码

| 代码 | 触发场景 |
| --- | --- |
| `E_MANIFEST` | JSON 损坏、版本不单调/重复、键非法、数值类型不支持、目标节点缺失、策略字段缺失等 |
| `E_CYCLE` | 派生依赖成环（独立业务码） |
| `E_GRID` | 精度口径 `grid_exp` 非法或并发预算非正整数 |
| `E_PORT` | 端口占用、连接失败、子进程退出未开放端口 |
| `E_HANDSHAKE` | 握手/下发真实套接字超时或错误应答 |
| `E_REPLY` | 回执缺字段、错误信封、连接中断、非法 JSON 行 |
| `E_CONVERGE` | 回执缺失或与权威重算不一致 |
| `E_PREVIEW` | 界面缓存与权威重算首个差异 |
| `E_REPLAY` | 快照重放首个不一致配置项键/字段 |

所有业务事件同步写入 `<workdir>/structured.log.jsonl`（JSON 行，含时间戳、事件名、
业务码与上下文字段）与 SQLite `ledger` 表。

## 实际验证结果

- `python -m unittest discover -s tests`：26 个用例全部通过（含真实子进程 TCP、
  端口占用、静默服务握手超时、缺字段回执、预算峰值、偏置节点 E_CONVERGE、
  快照篡改 E_REPLAY、预览漂移 E_PREVIEW）。
- `python -m configmesh.demo`：三版本流水线峰值并发 = 3 且不超过全局预算 3；
  三版全部收敛；v1→v2 仅 `base_temp` 直改并传播到 `heat_index/flow_score`，
  `base_pressure/base_flow` 产物逐字节不变；v2→v3 正确识别新增键、目标节点与
  分片策略变化；超时任务 attempt 1 为 E_HANDSHAKE、attempt 2 成功；快照身份不一致
  触发整体替换；`E_MANIFEST/E_CYCLE/E_GRID` 全部命中且注入后合法输入仍可处理。
- 验收产物：`workdir/acceptance/acceptance_report.json` 与
  `workdir/acceptance/acceptance_report.txt`，各子目录含结构化日志与独立 SQLite 账本。

## Windows 运行说明

- 直接使用 `python.exe` 即可，子进程通过 `sys.executable -m configmesh.protocol`
  启动；平台在 Windows 上自动附加 `CREATE_NO_WINDOW` 避免弹出控制台窗口。
- 若启用注册表中节点的固定重试，请同时将 `max_retries` 设为大于 0（强校验会拒绝
  `retry_policy=fixed` 但 `max_retries=0` 的配置）。
- 不使用 Docker，不访问回环以外的任何目标。
