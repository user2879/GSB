# 跨环境部署配置仲裁诊断工作台 — 技术说明

## 1. 目标与边界

工作台在单机 WSL Ubuntu 上首次落地，不使用容器或外部服务。每个部署环境由一个独立的
**子进程仲裁服务**在 `127.0.0.1` 回环端口上提供确定性仲裁；父进程通过真实 TCP 套接字
逐行 JSON 收发。样本由标准库 `random.Random` 按环境整数种子确定性生成，不访问网络或
外部数据集。SQLite（标准库 `sqlite3`）承担仲裁账本，WAL 模式为等待中的读者提供一致性
视图。

## 2. 模块结构

```
arbiter_workbench/
├── arbiter/
│   ├── errors.py          # 业务码与类型化异常
│   ├── spec.py            # JSON 声明加载与强校验 (E_SPEC)
│   ├── sample.py          # 确定性观测样本生成
│   ├── fixedpoint.py      # 定点量化与三段误差归因 (E_QUANT)
│   ├── protocol.py        # 逐行 JSON 套接字协议、超时、端口选择
│   ├── env_service.py     # 子进程环境仲裁服务（真实 TCP 监听）
│   ├── service_manager.py # 父进程子进程托管与回执
│   ├── merge.py           # 三类确定性合并口径
│   ├── planning.py        # 全量/增量显式规模选择规则
│   ├── propagation.py     # 失效传播与视图版本单调递增
│   ├── ledger.py          # SQLite 原子提交、摘要、快照整体替换
│   ├── recovery.py        # 崩溃恢复与 E_REPLAY 重放校验
│   ├── numpy_checks.py    # NumPy 独立向量化核对
│   ├── engine.py          # 主编排与 E_ORDER 确定性守卫
│   ├── gui.py             # Tk 诊断界面（无头降级为文本）
│   └── cli.py             # 命令行入口
├── decls/                 # 本地 JSON 声明与快照占位
├── tests/                 # 39 个 unittest 用例
└── scripts/               # WSL 环境包装、崩溃恢复演示
```

## 3. 业务要求到实现的映射

### 3.1 声明确载与强校验（要求 1，E_SPEC）
`spec.load_declarations` 强校验三类文件：环境（标识、回环 host 必须为 `127.0.0.1`、
端口 1024–65535、正整数租约容量、非负种子、正整数权重）、实例（实例标识、合法环境引用、
观测序号、毫秒时刻）、视图（视图标识、参与环境集合、每键口径仅限三种策略、非负整数容差）。
校验在**首个非法字段**处失败并通过 `ErrorContext(object_id, field, detail)` 报告对象与字段。

### 3.2 子进程服务与真实套接字（要求 2，E_PORT / E_TIMEOUT）
`env_service.EnvironmentServer` 在子进程内绑定真实回环监听套接字，协议为每行一个 JSON：
`HELLO` / `BATCH` / `ARBITRATE` / `SHUTDOWN`。父进程 `ServiceManager`：
- 先在父进程创建监听套接字并以 `pass_fds` 继承给子进程，消除“选号→绑定”抢占窗口；
- 保留端口带 `47100–47499` 内冲突自动换号重试，耗尽后报 `E_PORT`；
- 握手与读取均设置真实 `socket.settimeout`，超时统一报 `E_TIMEOUT`；
- 仲裁回执含环境标识、修订序号区间 `revision_min/max`、权威摘要 `authority_digest`
  与单调时钟毫秒 `elapsed_ms`。

### 3.3 三类确定性合并（要求 3，E_ORDER）
`merge.py` 实现：
- `latest`：修订序号最高优先，平局依次用毫秒时刻、环境标识；
- `weighted_majority`：按量化值分组累加环境整数权重，权重平局按取值升序稳定裁决；
- `interval_midpoint`：取最小/最大量化值的整数中点（向零截断），端点归属环境显式记录。

所有排序使用显式元组键，环境参与顺序固定为视图声明序，输出 JSON 一律 `sort_keys=True`。
`engine.ArbitrationEngine.assert_byte_identical` 在不同批次切分（batch=1/13/100）与不同
端口下逐字节比对，首个不稳定键以 `E_ORDER` 定位到视图与键。

### 3.4 全量/增量路径选择（要求 4）
`planning.decide` 使用纯整数成本模型：
- 全量：`E * K * N_avg`
- 增量：`A * K_dirty * N_avg + 2*(E+K)`
- 显式容差带：全量成本的 10%（整数向上取整）；带内按稳定规则选增量并记录容差影响；
  输出选择理由、关键规模参数与未触发判定的边界值。

### 3.5 失效传播（要求 5）
`propagation.PropagationEngine` 维护每视图单调递增版本号，沿 `env→view`、`env→key`、
`policy→key` 边推导。输出按视图标识序排序，每步带 `PropagationStep(edge, basis)`。
增量重算只处理 `dirty_keys`；测试验证未受影响键的 `KeyRecord` 逐字节保持不变。

### 3.6 定点量化与误差归因（要求 6，E_QUANT）
`fixedpoint.FixedPoint` 接受指数 0–3（量子 `10^-s`）：
- 截断方向：**向零截断**（`trunc_div_toward_zero`，对负数显式处理）；
- 溢出饱和：有符号 64 位 `[-2^63, 2^63-1]`，饱和被记录；
- 权重归一：最大余数法，归一权重之和严格等于 `10^s`；
- 误差界拆三段：量化截断（每操作数 <1 单位，线性组合上界为参与环境数）、
  区间边界归属（中点策略 1 单位）、权重归一（`n-1` 单位，仅加权多数）；
- 视图级聚合上界取逐键统一范数（各段最大值）。

### 3.7 原子账本与快照整体替换（要求 7，E_STATE）
`ledger.Ledger` 单事务写入一个视图的全部键记录（`BEGIN IMMEDIATE … COMMIT`，异常回滚）。
`meta` 携带整数 `format_version=1`、会话刻度与全库 `content_digest`（SHA-256，规范 JSON
序列化）。快照恢复是**整体替换**：先清空两表再导入，摘要不符报 `E_STATE`，绝不增量合并。
WAL + `synchronous=FULL` 让并发读者始终看到已提交的一致性视图。

### 3.8 崩溃恢复与重放（要求 8，E_REPLAY）
中断事务只留下 `UNKNOWN` 标记行。`recovery.recover` 仅信任完整提交记录：UNKNOWN 视图
进入 `unknown_views` 要求重跑，且明确不生成强制终止退出报告
（`assert_no_fabricated_exit_report`）。对已提交视图，用相同确定性输入重放并逐键逐字段
比较权威取值、采纳/被否集合与误差归因，首个差异以 `E_REPLAY` 定位
`(view_id, field, key)`。

## 4. 业务码核对

| 码 | 触发位置 | 含义 |
|----|----------|------|
| E_SPEC | `spec.py` | 首个非法声明字段/对象 |
| E_PORT | `protocol.py` / `service_manager.py` / `env_service.py` | 端口占用、连接拒绝、重试带耗尽 |
| E_TIMEOUT | `protocol.py` | 握手或读取真实超时、服务未就绪 |
| E_ORDER | `engine.py` | 跨顺序/批次输出不一致 |
| E_QUANT | `fixedpoint.py` | 非法刻度、除零、非法权重 |
| E_STATE | `ledger.py` / `recovery.py` | 快照版本/摘要不一致、格式不匹配 |
| E_REPLAY | `ledger.verify_replay` | 重放首个不一致视图/键/字段 |
| E_INTERNAL | `errors.py` | 防御性内部不变量失败 |

## 5. 验证结果

- `python3 -m unittest discover -s tests`：**39 个用例全部通过**（退出码 0）。
- 覆盖：声明非法字段、三类口径顺序无关性、定点截断/饱和/三段误差、路径选择容差带、
  失效传播与未污染键保持、真实套接字 E_PORT/E_TIMEOUT、跨批次逐字节一致、
  快照整体替换与篡改拒绝、UNKNOWN 恢复与 E_REPLAY 定位、NumPy 向量化核对。
- `scripts/crash_recovery_demo.py`：完整演示强制终止→UNKNOWN 隔离→重放校验→
  快照导出/篡改拒绝，末行打印 `CRASH_RECOVERY_DEMO_OK`。
- Tk：有显示时真实创建 Tk 窗口；无显示时 `gui.py` 自动降级为无头文本报告。
