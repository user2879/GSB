# 推理网关路由诊断台（本地 Windows 目标环境）

面向本地环境的推理网关路由诊断组件：声明式后端/规则加载与强校验、确定性
指纹与分批计划、稳定哈希灰度分流、双预算并行调度、SQLite 命中账本与强制
终止恢复重放、定点统计与配置演进陈旧防护、Tkinter 诊断界面。

## 环境

- Python 3.11/3.12，NumPy，Tkinter（官方 Windows 安装包自带），sqlite3。
- 无 Docker、无外网、无 GPU；样本回环端口 18081-18084 已验证可监听。

## 结构

- `gateway/loader.py` — JSON/NDJSON 加载与强校验，报错带字段路径（如 `backends[1].port`）。
- `gateway/fingerprint.py` — 指纹 = sha256(model_id | 量化参数(6 位小数) | 参数长度 | 配置摘要 | 规范加载序号)，加载序号为按 model_id 排序后的秩，与文件顺序/并发无关。
- `gateway/routing.py` — sha256(rule_id#client_key) 取桶 [0,10000)，与灰度权重万分比比较；最长前缀匹配规则，rule_id 决胜。
- `gateway/batching.py` — 三种确定性分批策略与规模自适应选择；平局阈值 `STRATEGY_TIE_THRESHOLD=1`（批），按策略名字典序稳定决胜。
- `gateway/scheduler.py` — 全局信号量 + 每后端信号量双预算；单调时钟毫秒记录排队等待；超时重派不改变输入集合与输出摘要。
- `gateway/ledger.py` — 每批次单事务写入批次行 + 逐请求账本行 + 进度游标；携带整数格式版本与内容摘要；配置哈希（权重/批大小/配额/策略）变化时版本单调递增。
- `gateway/replay.py` — 恢复扫描（已提交/UNKNOWN）与重放校验，首个不一致以 `E_REPLAY` 报出请求序号与字段。
- `gateway/stats.py` — 万分比与平均耗时的整数定点表示，floor 舍入，误差上界 1bp / 1ms，同时携带精确 (分子, 分母)。
- `gateway/ui.py` — Tkinter 界面；`build_view_model` 为纯函数，渲染顺序稳定，可无头测试。
- `samples/` — 确定性样本（种子 20261121）、三场景演示 `demo.py`、强制终止子进程 `crash_runner.py`、UI 启动 `run_ui.py`。
- `tests/` — 25 个 pytest 用例。

## 策略自适应规则（确定性）

1. 后端数 ≥ 3 且配额极差 > 2 倍 → 倾向 `quota_spread`（按配额份额缩放各后端批大小）；
2. 平均每客户端请求数 ≥ 2×批大小 → 倾向 `client_cluster`；
3. 否则 `arrival_fill`。
对全部候选策略估算工作量（批数），与倾向策略之差 ≤ 阈值 1 批的候选进入
平局集合，按名字字典序稳定决胜；选择理由与关键规模参数随结果输出。

## 运行

```bash
python samples/gen_samples.py   # 重新生成确定性样本
python samples/demo.py          # 三场景演示（小规模 / 扩大规模+崩溃恢复 / 权重变更）
python samples/run_ui.py        # Tkinter 诊断界面（需显示环境）
python -m pytest tests/ -q      # 测试
```

## 验证结果（本机实测）

- `python -m pytest tests/ -q`：25 passed。覆盖强校验字段路径、指纹/分批/
  路由在乱序声明下的不变性、双预算上限不越界、超时重派摘要不变、崩溃恢复
  重放、E_REPLAY 篡改定位、定点误差界、版本单调递增与陈旧排除、视图模型
  稳定顺序。
- `python samples/demo.py`：
  - 场景 1（8 请求）：三策略工作量平局（3/3/3），稳定决胜 `arrival_fill`；
    峰值并发 2，未超全局上限。
  - 场景 2（96 请求）：子进程提交 5 批后以退出码 137 模拟强制终止；恢复
    扫描正确列出 5 个已提交批与 27 个 UNKNOWN 批，游标与内容摘要一致；
    续跑跳过已提交批，重放校验 `compared_requests=96, compared_batches=32,
    status=OK`；灰度占比 r-chat 2656/10000（目标 2500，floor 舍入，
    误差界 ±1bp），峰值并发 4、各后端峰值均未超配额。
  - 场景 3（权重 2500→6000）：配置版本 1→2 单调递增，v1 的 96 行账本标记
    陈旧且不并入当前报告（整体替换），当前版本重放 OK，快照列表含两个版本。
- 回环端口 18081-18084 绑定监听成功。

注：本开发容器为 Linux 且无 root，无法安装 tkinter，故 GUI 以
`py_compile` + 无头视图模型测试验证；在 Windows 官方 Python（勾选
tcl/tk）环境下 `python samples/run_ui.py` 直接运行。
