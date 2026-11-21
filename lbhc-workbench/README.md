# LBHC — 负载均衡健康检查诊断工作台

面向本地环境（Windows / Linux / macOS）的纯 Python 3.11 项目：从本地 JSON 声明
负载均衡集群，在真实套接字上执行 TCP/HTTP 健康探测，在全局并发预算、每池并发
配额与全局速率上限三重约束下调度，维护每实例健康状态机与判定理由，并把每次
探测、状态迁移与预算消耗写入本地 SQLite 账本。不依赖 Docker 或任何外部服务，
仅访问 127.0.0.1。

## 目录结构

```
lbhc/            工作台包
  spec.py        声明加载与强校验（E_SPEC，字段路径定位）
  backend.py     模拟后端子进程（ok / mismatch / delay / refused 四类行为）
  probe.py       真实套接字 TCP/HTTP 探测
  scheduler.py   预算+配额+速率三重约束调度器（单派发线程保证序列确定）
  paths.py       全量 / 按池分层抽样两条探测路径与自适应选择
  health.py      unknown/healthy/unhealthy/draining 状态机
  ledger.py      SQLite 账本（单事务提交、inflight、sha256 哈希链）
  replay.py      重放一致性校验（E_REPLAY，定位首个分歧实例与字段）
  engine.py      运行编排：端口预检（E_PORT）、后端拉起、恢复、报告
  ui.py          Tkinter 诊断界面（只读账本，稳定渲染顺序）
  cli.py         命令行入口与三类异常注入
data/cluster.json     确定性样本声明（端口 18601-18610）
demo/run_demo.py      四场景可重复演示
tests/test_workbench.py  23 项单元/端到端测试
scripts/check_ports.py   预留端口占用检查
scripts/start_backend.py 手工启动单个模拟后端（调试用）
```

## 快速开始

```bash
python --version          # 需要 3.11；numpy 已用于统计，tkinter 为界面依赖
python scripts/check_ports.py
python -m lbhc.cli gen-spec --out data/cluster.json
python -m lbhc.cli run --spec data/cluster.json --ledger out/run.db \
    --rounds 4 --path auto --report out/run-report.json
python -m lbhc.cli verify --spec data/cluster.json --ledger out/run.db
python -m lbhc.cli ui --ledger out/run.db        # 打开 Tkinter 诊断界面
python demo/run_demo.py                          # 四场景演示
python -m unittest discover -s tests -v          # 测试
```

## 关键设计

**真实判定**：成功/失败只来自真实套接字——超时由 `socket.settimeout` 到期
产生，连接拒绝由真实连接错误产生（`refused` 后端以 `SO_LINGER(1,0)` 立即
RST，或对未监听端口 connect 得到 ECONNREFUSED），摘要不符来自对响应字节
重新计算 sha256。

**调度**：单个派发线程按确定性顺序（pool、instance 排序）遍历任务；依次获取
全局预算信号量、池配额信号量，最后获取速率令牌（滑动 1 秒窗口）才记录
`started_ms` 并启动工作线程。排队开始/结束均取单调时钟毫秒。同一声明多次
运行得到相同的发起实例序列（测试 `test_deterministic_initiated_sequence`）。

**双路径一致**：分层路径只对已达到稳定判定（阈值达成）的实例降频，且每 3 轮
回补一次；降频实例不参与探测，沿用上次判定，不计成功/失败。由于稳定实例在
确定性后端下不再产生迁移，两条路径输出逐字节一致的健康判定与迁移序列
（`test_dual_path_byte_identical`）。路径自适应选择报告实例数、池数与预算
余量。

**账本事务**：每条探测结果 + 状态迁移 + 实例视图 + 进度游标 + 哈希链事件在
同一 SQLite 事务提交；派发前先提交 `inflight` 行。进程被 kill -9 后，遗留
inflight 即“未提交探测”，恢复时标记为 unknown（`probe_void` 事件）并要求
重跑；被杀的运行不会写出退出报告。`replay_verify` 依据账本中的声明版本与
探测序列重放状态机，逐项比对视图与迁移序列，首个分歧以 E_REPLAY 报出实例
标识与字段。events 表 sha256 哈希链检测账外篡改（E_LEDGER）。

**陈旧防护**：声明版本单调递增（回退以 E_SPEC 拒绝）。版本演进时旧版本视图
整体标记 `stale=1`，当前视图整体替换（非增量合并），并稳定列出状态/判定
理由变化的实例（added/removed/updated）。

**异常注入**（写入 `logs/lbhc-*.jsonl` 结构化日志，退出码=业务码）：

```bash
python -m lbhc.cli inject spec-corrupt   --spec data/cluster.json --ledger out/i1.db  # E_SPEC=2
python -m lbhc.cli inject port-occupied  --spec data/cluster.json --ledger out/i2.db  # E_PORT=3
python -m lbhc.cli inject ledger-digest  --spec data/cluster.json --ledger out/i3.db  # E_LEDGER=4
```

## 实际验证结果（本机，Python 3.11）

- `python -m unittest discover -s tests`：23/23 通过（约 18s）。
- `python demo/run_demo.py`：四场景全部通过——
  1. 全量路径：峰值并发 4/4，池占用 http 3/3、tcp 2/2，未超预算；
  2. 分层路径：探测数更少（33 vs 36），判定与迁移序列逐字节一致；
  3. 预算变更（版本 1→2）：旧版本 10 行视图标记 stale，当前视图整体替换，
     diff 稳定列出 h5 removed、h7 added；
  4. kill -9（退出码 -9）：14 条已提交、4 条 inflight 遗留、无退出报告；
     恢复后重放校验通过（54 probes / 9 transitions / 10 instances）。
- 四类判定理由均由真实套接字产生：threshold_met / timeout /
  conn_refused / digest_mismatch。

## 备注

- 本工作台在 Linux 容器开发验证；目标 Windows 环境自带 tkinter。若解释器
  无 tkinter，`ui` 子命令会给出明确提示而非崩溃。
- 模拟后端通过 stdin 管道看门狗随父进程退出，即使父进程被 kill -9 也不会
  残留占用端口（POSIX 与 Windows 均有效）。
