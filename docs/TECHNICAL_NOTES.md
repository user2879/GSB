# 技术说明：模型热更新灰度快照重建与恢复一致性验证组件

## 1. 范围与约束

- 纯本地、纯标准库 + NumPy 实现；无真实模型文件、无外部账号、无线上服务。
- 后端为独立 Python 子进程，仅监听声明文件显式给出的 `127.0.0.1` 高位端口。
- 通信协议为 TCP 上的 JSON 行（每条消息一个 JSON 对象 + `\n`）。
- 所有样本、参数、输入均由整数种子的显式规则生成，可逐位复现。
- 不使用 Docker；SQLite 走标准库 `sqlite3`，Tkinter 由 Windows 官方 Python 自带。

## 2. 包结构

| 模块 | 职责 |
| --- | --- |
| `hotgray/errors.py` | 业务错误码 `E_SPEC/E_PORT/E_HANDSHAKE/E_REPLY/E_SNAPSHOT/E_REPLAY/E_STATE` |
| `hotgray/hashing.py` | 规范化 JSON、SHA-256 内容摘要、稳定哈希分桶与灰度判定 |
| `hotgray/determinism.py` | 确定性参数矩阵、输入向量、量化前向推理、参数/输入/输出摘要 |
| `hotgray/spec.py` | 声明文件加载与强校验，整体拒绝并给出首个违规字段路径 |
| `hotgray/protocol.py` | JSON 行帧封装、真实套接字超时、回执必填字段校验 |
| `hotgray/backend.py` | 后端子进程入口与网关侧进程控制器（同连接热替换、版本保留） |
| `hotgray/storage.py` | SQLite 账本/批次/参数版本/重建轨迹/会话快照，行级内容摘要与事务 |
| `hotgray/snapshot.py` | 版本链归并、快照摘要、权威重算、逐字段比对、恢复重放 |
| `hotgray/gateway.py` | 分流、按批拆分、热替换协调、独立线程重建与不相交保证 |
| `hotgray/gui.py` | Tkinter 快速视图（延迟导入）与权威重算按钮 |
| `hotgray/cli.py` | `validate/run/recompute/replay/inject/gui` 命令行入口 |
| `hotgray/samples.py` | 确定性声明样本（含一个非法样本）生成器 |

## 3. 关键设计与业务要求映射

### 3.1 声明强校验（要求 1）
`spec.validate_declaration` 按 JSON 风格路径（如 `backends[1].port`）逐项校验：
标识唯一、端口必须为字面量 `127.0.0.1` 且全局不重复、维度为正整数、灰度权重
在 `[0,10000]`、规则引用的后端必须存在。任一违规立即抛 `E_SPEC`，整份声明拒绝加载。

### 3.2 确定性构造（要求 2）
- 参数：`hotgray/determinism.py` 用 `sha256(seed|tag|row|col) % 9973` 显式生成
  量化整数矩阵 `w1/b1/w2/b2`，不依赖 NumPy RNG 的版本行为。
- 参数摘要绑定 `backend_id + model_version + quant_version + 维度 + seed + 各矩阵摘要`。
- 热替换种子由 `next_hotswap_seed(seed, seq)` 显式派生；新版本摘要取代当前版本，
  旧版本在子进程与 `param_versions` 表中保留为只读对照。
- 输入向量由 `sha256(rule|sample|col)` 生成并缩放至 `[-1,1)`，输入摘要同样可复现。

### 3.3 子进程编排与热替换（要求 3）
- 网关以子进程方式启动 `hotgray/backend.py`，等待 `ready` 行后建立 TCP 连接并 `ping` 握手。
- 所有 `connect/recv/send` 均设置真实超时；端口占用为 `E_PORT`，握手/切换超时为
  `E_HANDSHAKE`，回执缺字段或样本集合不符为 `E_REPLY`。
- 热替换在同一连接上发 `hotswap`，后端递增版本序号并保留旧参数，不重建连接。
- 回执包含模型版本、参数摘要、批内样本集合、逐样本输出摘要、整批摘要与整数毫秒耗时。

### 3.4 稳定哈希分流与命中账本（要求 4）
- `stable_bucket = int(sha256(canonical([rule_id, client_key]))[:8], 16) % 10000`。
- `bucket < gray_weight_bp` 命中灰度，否则稳定版；同键同规则同权重结果恒定。
- 账本逐请求记录规则、去向、桶、权重、批次、子批、后端版本序号、参数摘要、
  输入/输出摘要与耗时；热替换前后命中通过 `backend_seq + param_digest` 区分版本。

### 3.5 版本链归并与拆批（要求 5）
- 快照以 `(rule_id, destination)` 为维度，按 `backend_seq` 升序沿版本链归并。
- 同一批内若出现两个参数摘要版本，`_split_by_version` 在版本边界显式拆为子批，
  每个子批只含单一 `backend_seq`，并写入 `split_reason`，杜绝静默混批。
- 快照登记账本版本区间、参与归并的参数摘要版本集合、输出摘要链与快照内容摘要。

### 3.6 重建期间准入与不相交（要求 6）
- 重建在独立线程执行，开始时冻结账本上界 `ledger_upper_bound`。
- 新请求继续走当前权重准入并记账，账本版本可继续递增。
- 快照只纳入 `[1, upper_bound]` 的命中，并登记
  `incoming_during_rebuild = 当前最大序号 - upper_bound`，作为归属区间复核依据。

### 3.7 持久化、强制终止恢复与会话快照（要求 7）
- SQLite 表 `hits/batches/param_versions/rebuilds/snapshots/batch_intent/meta`，
  格式版本固定为整数 `1`，每条命中携带自身内容摘要。
- 批次先写 `batch_intent`，命中行、批次行与进度游标在同一事务提交后删除意图行。
- 强制终止后重启：已提交事务可恢复；残留意图行标记为
  `unknown_rerun_required` 并要求重跑，系统不生成任何“退出报告”。
- 会话快照写入完整账本区间、参数版本集合与快照摘要；重开时若版本或摘要不一致，
  `replace_all_snapshots` 整体替换，不做增量合并。

### 3.8 快速视图/权威重算分离、重放与异常注入（要求 8）
- Tkinter 快速视图只读快照统计（灰度占比、批命中、耗时、版本链、拆批）。
- 权威重算按账本逐请求重放：稳定哈希分流 → 由持久化种子重建参数并核对参数摘要 →
  重建输入向量 → NumPy 前向 → 逐样本输出摘要 → 重新归并；与快照逐字段比较，
  首个差异以 `E_SNAPSHOT` 报出规则与字段。
- 恢复重放按快照记录的账本区间与参数版本集合重新归并，不可复现以 `E_REPLAY`
  报首个不一致项；行内容摘要不符同样为 `E_REPLAY`。
- `hotgray inject` 支持三类注入：`corrupt_reply`（E_SNAPSHOT）、
  `missing_param_version`（E_REPLAY）、`ledger_digest`（E_REPLAY）。

## 4. 浮点确定性说明

输出摘要在 `np.tanh` 后统一 `round(...).astype(int64)` 再参与哈希，避免把浮点字节
直接写入摘要。在同一 NumPy 主版本与同一 CPU 架构（Windows x64 / WSL2 x64）下，
权威重算与后端子进程得到逐字节一致的整数摘要。耗时字段为运行时测量，仅用于统计，
不进入任何内容摘要，因此不影响逐字节比对。

## 5. Windows 10/11 操作步骤

```powershell
# 1. 创建虚拟环境并安装依赖（Tkinter 与 sqlite3 随官方 Python 自带）
powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1

# 2. 生成确定性样本并校验
python -m hotgray.samples --out-dir samples
python -m hotgray validate samples/gateway_declaration.json

# 3. 端到端 / 强制终止恢复 / 异常注入
python scripts/demo_e2e.py
python scripts/demo_recovery.py
python scripts/demo_faults.py

# 4. CLI 单步运行与权威重算、恢复重放
python -m hotgray run samples/gateway_declaration.json --db runtime/demo.sqlite3 --hotswap-after 10
python -m hotgray recompute samples/gateway_declaration.json --db runtime/demo.sqlite3
python -m hotgray replay samples/gateway_declaration.json --db runtime/demo.sqlite3

# 5. Tkinter 诊断界面
python -m hotgray gui samples/gateway_declaration.json --db runtime/demo.sqlite3
```

Linux/WSL 下使用 `scripts/setup_env.sh`；若 WSL 未带 Tk，界面命令返回退出码 3 并提示
安装 `python3-tk`，其余全部功能不受影响（本环境即为此情形）。

## 6. 实际验证结果（本工作区 WSL2 / Python 3.11 / NumPy 1.26.4）

- 单元 + 集成测试：`python -m unittest discover -s tests` → **25 passed**。
- `demo_e2e.py`：24 次命中、2 个灰度后端热替换到 seq=1、记录 11 处版本边界拆批、
  快照与权威重算 `byte_identical`。
- `demo_recovery.py`：阶段一进程被 SIGKILL（returncode -9），阶段二仅恢复 8 条已
  提交命中，`batch-ORPHAN` 标记为 `unknown_rerun_required`，补齐请求后重放
  `reproducible`。
- `demo_faults.py`：三类注入分别得到 `E_SNAPSHOT`（定位到具体规则/样本/字段）、
  `E_REPLAY`（参数版本缺失）、`E_REPLAY`（账本内容摘要不符）。
- 确定性：同一份声明重复构造，4 个后端参数摘要完全一致。
