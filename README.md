# 推理网关灰度回执归并诊断台

面向本地 Windows 的确定性灰度诊断应用。项目从空白工作区构建，默认只访问 `127.0.0.1`，不发起任何外部网络请求，不依赖 Docker。

核心能力：

- 本地 JSON 强校验后端、路由规则与 NDJSON 请求样本，非法声明整体抛出 `E_SPEC` 并定位首个字段。
- 固定种子的 SplitMix64 + NumPy 前向推理；参数、输入与输出摘要可逐位复现。
- h1 / h2 / h3 历史回执编解码、未知字段保留、跨热更新版本归并。
- 客户端键稳定哈希分桶、声明批大小、并发/乱序提交下规范化批次划分。
- 整数定点耗时与平均样本数、单值误差界、批次聚合误差界和误差内稳定决胜。
- SQLite 单事务批次账本、整数格式版本、内容摘要、逐批游标。
- 会话快照快速视图与账本权威重算彻底分离，支持 `E_PREVIEW` / `E_REPLAY`。
- Tkinter 交互式诊断台，以及真实回环 TCP JSONL 网关 + 后端子进程。

## 快速开始

环境要求：Python 3.10+、NumPy、标准库 `sqlite3` 与 `tkinter`。Windows 官方 Python 安装器默认包含 Tk；若缺失，请在安装器中勾选 “tcl/tk and IDLE”。

```bat
py -3 -m pip install -r requirements.txt
run_doctor.bat
run_demo.bat
run_tests.bat
run_app.bat
```

无界面环境可直接运行：

```bash
python3 -m gray_merge_lab doctor --spec
python3 -m gray_merge_lab demo --out artifacts/demo
python3 -m unittest discover -s tests -v
```

演示输出位于 `artifacts/demo/`：

- `ledger.sqlite3`：权威批次账本。
- `snapshot.json`：快速视图快照。
- `checkpoint.json`：中途检查点，用于恢复重放。
- `receipts.ndjson`：全部 h1/h2/h3 回执。
- `report.json`、`REPORT.md`：本次实际验证结果。

## 实际验收结果

当前环境执行 `python3 -m gray_merge_lab demo --out artifacts/demo --scale-exp 3` 的结果：

```text
DEMO OK
  three_format_merge: PASS
  route_stability: PASS
  fixedpoint: PASS
  recovery_replay: PASS
  authoritative_recompute: PASS
  anomaly_injection: PASS
  receipt_counts: {'h1': 8, 'h2': 8, 'h3': 9}
  fixed_scale_exp: 3
  bounds: single<0.001 combined_q=2
  recovery: after 2 replayed 6 batches
```

`python3 -m unittest discover -s tests -v`：**74 项全部通过**，包含：

- 强校验首个违规字段、重复标识/端口、维度、量化口径、权重范围。
- 参数/输入/输出逐位重复一致，参数摘要绑定后端、维度、口径与版本。
- h1/h2/h3 编解码、未知字段、摘要格式、重复归并序号、跨版本归并、样本差异。
- 分流稳定性、乱序/并发批次一致性、定点误差界与候选稳定决胜。
- SQLite 事务回滚、内容摘要篡改检测、快照整体替换与 `E_SNAPSHOT`。
- 权威重算 `E_PREVIEW`、恢复重放 `E_REPLAY`、三类异常注入。
- 真实 `127.0.0.1` 后端子进程 + 网关 JSONL 集成，热更新跨进程同步。

本构建容器缺少操作系统 Tk 共享库，因此 `tkinter` 运行验证不可用；核心、CLI、网关、账本与全部非 GUI 测试均已实际运行。界面层已完成编译检查，Windows 安装了 Tk 后可用 `run_app.bat` 启动。

## 目录结构

```text
gray_merge_lab/
  data/                 本地后端、规则、样本与热更新声明
  specs.py              强校验与版本摘要
  deterministic.py      SplitMix64、参数/输入构造、NumPy 前向
  routing.py            稳定哈希、分流、规范化批次
  receipts.py           h1/h2/h3 编解码
  merge.py              跨版本稳定归并
  fixedpoint.py         定点统计、误差界与候选决胜
  ledger.py             SQLite 事务账本
  snapshot.py           快照、版本识别与整体替换
  replay.py             权威重算与恢复重放
  engine.py             分流、推理、回执、账本编排
  gateway.py            回环 JSONL 网关与 TCP 后端传输
  backend_service.py    回环推理后端子进程
  viewmodel.py          GUI 无关的会话控制器
  gui.py                Tkinter 诊断台
  demo.py               可重复端到端演示
tests/                  74 项标准库 unittest
docs/                   设计与协议说明
scripts/                网关样本发送工具
artifacts/demo/         实际验证产物
```

## 关键口径

- 分流桶：`SHA-256` 规范化 JSON `{rule_id, client_key}` 的前 64 位模 10000 后加 1，得到 1..10000；`bucket <= weight_bp` 走灰度，权重 0/10000 严格对应全稳定/全灰度。
- 批次：按规则、去向、客户端键规范化后，再按声明批大小切块；客户端同质保证历史 h2 的单桶值语义准确。显式 `flush` 定义一个确定性的接收纪元。
- 归并：先取最高 `merge_seq`，同序号取最高参数版本，最终用回执摘要字典序稳定决胜；相同 `(batch_id, merge_seq)` 直接 `E_FORMAT`。
- 定点：`S=10^scale_exp`；正值除法向零截断。批内整数耗时/样本数的缩放值精确，聚合均值单值误差 `<1/S`；独立聚合 N 批误差上界 `<N/S`；先累加原始整数再截断的规范聚合误差 `<1/S`。
- 候选：两个定点平均耗时差 `<=2` 个定点单位时视为受误差影响，按批次标识字典序稳定决胜并标注。
- 快照：后端集合、规则、样本或缩放口径任一变化即整体替换，并抛出带稳定差异列表的 `E_SNAPSHOT`。

详细公式、账本结构、错误码与 JSONL 协议见 [docs/DESIGN.md](docs/DESIGN.md) 与 [docs/PROTOCOL.md](docs/PROTOCOL.md)。

## 网关演示

终端 1：

```bat
run_gateway.bat
```

终端 2：

```bat
send_samples.bat 28190 gray_merge_lab\data\samples.ndjson
```

网关在 `127.0.0.1:28190` 接收 JSONL，后端子进程分别绑定声明端口 `28101..28104`。发送工具逐行发送样本后发送 `{"op":"flush"}`，返回批次、去向、输出摘要与参数摘要。

## 诊断台界面

`run_app.bat` 启动后：

- “运行总览”只读 `snapshot.json`，显示每条规则的稳定/灰度批与样本分布、批命中、原始耗时和定点统计。
- “归并详情”展示对比顺序、采纳/否决、否决理由、样本差异与未知字段候选。
- “定点与权威复核”在点击“权威重算”前保持灰色；成功后显示账本逐批重放结论。
- “账本与快照”显示 SQLite 内容摘要、快照摘要、manifest、进度与环境。
- “异常注入”覆盖摘要损坏、参数字段缺失、快照摘要不符、预览差异与重放差异。
