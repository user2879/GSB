# 设计与技术说明

## 1. 目标与边界

本组件在 Windows 本地回环环境（开发与 CI 亦可在 Linux/macOS 运行，仅进程挂起原语不同）
实现确定性的混沌实验。所有依赖均为 Python 3.11+ 标准库：`tkinter`、`sqlite3`、
`subprocess`、`socket`、`threading`、`ctypes`。不访问外网、不使用容器、不需要第三方包。

四条硬约束贯穿设计：

1. 注入只改变**运行时输入参数**或**目标进程状态**，绝不修改任何源码文件。
2. 同一声明 + 同一整数种子，重复执行产生逐字节一致的注入时间轴摘要与可比较的观测结论。
3. 强制终止后的恢复只依赖已完整提交的 SQLite 事务，不承诺一定生成退出报告。
4. 任一校验/摘要/重放失败都用稳定业务码（`E_SPEC`/`E_TIMELINE`/`E_TARGET`/
   `E_LEDGER`/`E_REPLAY`）整体拒绝，并定位到具体字段。

## 2. 模块结构

| 模块 | 职责 |
| --- | --- |
| `chaoskit/spec.py` | 声明加载与强校验，全量收集字段路径，失败整体拒绝（`E_SPEC`） |
| `chaoskit/rng.py` | 自研 SplitMix64 整数 PRNG 与无状态 `mix_u64`，跨平台逐位一致 |
| `chaoskit/timeline.py` | 稳定键排序 + 种子驱动的时间轴、摘要与首偏差校验（`E_TIMELINE`） |
| `chaoskit/mock_service.py` | 确定性回环目标服务（真实子进程，行协议，按种子规则拒绝请求） |
| `chaoskit/process_control.py` | Windows `NtSuspendProcess`/`TerminateProcess` 与 POSIX 信号 |
| `chaoskit/target.py` | 子进程托管、孤儿回收、暂停/终止状态区分、连接层延迟代理 |
| `chaoskit/observe.py` | 单调时钟采样、逐次请求记录、定点成功率/分位数与量化误差界 |
| `chaoskit/ledger.py` | SQLite 账本、单事务提交、unknown 标注、版本/摘要识别与整体替换 |
| `chaoskit/engine.py` | 编排器：生命周期、四种注入、三阶段观测、恢复续跑 |
| `chaoskit/replay.py` | 身份/结构/观测三级重放校验与字段级差异定位（`E_REPLAY`） |
| `chaoskit/viewmodel.py` | 无 Tk 依赖的纯数据视图模型，渲染顺序稳定、可单测 |
| `chaoskit/gui.py` | Tkinter 诊断界面（懒加载，无头环境用 `render_headless`） |
| `chaoskit/cli.py` | `validate/timeline/run/recover/replay/gui` 命令行入口 |

## 3. 确定性是怎么保证的

### 3.1 随机源

不使用宿主 `random` 模块（其流在不同 Python 版本间变化且可能引入 OS 熵）。
`rng.SplitMix64` 是基于计数器的 64 位整数生成器，同种子在任何平台输出同一整数序列；
`next_below` 用拒绝采样消除取模偏差。

### 3.2 时间轴与书写顺序无关

注入项按稳定键 `(target_id, injection_type, injection_id)` 排序后再消费随机流，
因此 JSON 中注入项的书写顺序不影响结果。每个条目的开始时刻、目标、类型、时长、
参数被规范化为排序键 JSON 后做 SHA-256，得到逐字节稳定的时间轴摘要。
`examples/sample_experiment.json` 与 `sample_experiment_shuffled.json` 同语义但乱序，
两者摘要完全相同（演示场景 1 断言）。

### 3.3 服务侧错误判定可复现

错误比例注入不是让服务“随便失败”，而是下发 `(seed, seq, error_bps)` 运行时规则。
每个请求用无状态的
`SplitMix64(mix_u64([seed, seq, request_index, "error_rate"])).next_below(10000)`
决定接受/拒绝。请求序号按服务生命周期内单调计数，重放同一序列得到逐位一致的成败结果，
因此**成功率结论（成功数、失败数、bps）跨运行严格相等**。

### 3.4 哪些量严格相等，哪些量带容差

* 严格逐位一致：时间轴摘要、注入结构字段、每个请求的成败、成功率定点值与样本计数。
* 仅在量化/调度容差内比较：毫秒级延迟分位数。线程调度、socket 缓冲与代理转发存在
  物理抖动，无法逐位承诺；延迟以整数毫秒记录，量化误差上界 1ms，重放容差另加一个采样
  间隔的调度包络。结构性漂移（如修改 `fixed_cost_ms`、注入时长）远在该包络之外，仍会被
  `E_REPLAY` 可靠捕获。

## 4. 四种真实注入机制

| 注入 | 真实机制 | 改变的对象 | 回滚 |
| --- | --- | --- | --- |
| `network_delay` | 编排器进程内的回环 TCP 转发代理在转发前 `sleep(delay_ms)` | 代理运行时延迟旋钮 | 延迟置 0 |
| `error_rate` | 通过控制通道向服务下发种子化拒绝规则 | 服务运行时输入参数 | 下发 `CLEAR` |
| `suspend` | Windows `NtSuspendProcess`；POSIX `SIGSTOP` | 目标进程真实挂起 | `NtResumeProcess`/`SIGCONT` |
| `terminate` | Windows `TerminateProcess`；POSIX `SIGKILL` | 目标进程被强杀 | 重新拉起子进程 |

服务不可达或端口被未知进程占用统一报 `E_TARGET`。监听端口可连接并不足以区分挂起
（已握手的 backlog 在内核中仍在），因此可达性判定要求拿到应用层 `PING/READY` 响应。
账本 `observed_state` 记录注入生效期间的真实状态，可明确区分 `suspended` 与 `terminated`，
即使回滚后进程已被恢复或重启。

## 5. 观测口径与定点量化

* 时钟基准：`time.monotonic_ns() // 1_000_000`，本地单调计数毫秒，不使用墙钟，不受系统时间回调影响。
* 真值来源：账本 `request_records` 表中的逐次请求记录（时间戳、阶段、成败、延迟、错误）。
  汇总只是派生值。
* 采样口径：每注入 `baseline/active/post` 三阶段，固定采样间隔与固定样本数，随账本持久化
  （`observations.sample_interval_ms` 与 `sample_count`）。每阶段样本数固定，保证请求序列与
  种子化成败决策跨运行对齐；注入时长仍真实控制故障持续时间。
* 成功率定点：`success_rate_bps = floor(success*10000/n)`，量化误差上界
  `ceil(10000/n)` bps，随每条摘要持久化。
* 延迟分位数：最近秩（nearest-rank）p50/p95/max，整数毫秒，量化误差上界 1ms。
* 样本不足：`sample_count < min_samples` 时 `insufficient=true`，派生指标置 `null`，
  保留原始成功/失败计数，绝不伪造数值。

## 6. 账本、事务与强制终止恢复

SQLite 启用 WAL 与 `synchronous=FULL`。

* `mark_running` 与 `commit_injection` 各自是一个完整事务，且**每个改变内容的事务都刷新
  `meta.content_digest`**，保证硬杀发生在任意时刻，磁盘上的摘要与内容都自洽。
* 单条注入提交事务同时写入：注入行、三阶段观测摘要、全部逐次请求记录、进度游标。
  因此硬杀后只有“完整提交”或“什么都没有”两种结果。
* 重开顺序：先校验 `meta` 内容摘要（失败报 `E_LEDGER`），再把残留 `running` 重分类为
  `unknown` 并要求重跑。恢复报告列出已恢复注入数、unknown 序号、待办序号与剩余预算，
  并显式 `exit_report_promised=false`——本组件不承诺一定能生成退出报告。

## 7. 版本识别与整体替换

账本 `experiment` 表携带账本格式版本、声明版本、种子、声明摘要与时间轴摘要。重开时若
快照中的格式版本、声明版本、种子或声明摘要与当前会话不一致，执行**整体替换而非增量合并**：

1. 旧库复制为 `ledger.backup-<UTC时间戳>.sqlite3` 并 `chmod 0o444` 只读（含 `-wal/-shm`）。
2. 删除旧库并按当前会话重新初始化注入集合、观测与进度。
3. 在新库 `replacements` 表记录新旧格式版本、两份声明摘要与跳过的已提交注入数，
   界面与报告按稳定顺序列出。

## 8. 重放校验与差异定位

`replay.py` 分三级，全部失败以 `E_REPLAY` 报明依据：

1. 身份校验：种子变化或服务配置摘要不一致，定位到 `seed`/`spec_digest`/`timeline_digest`。
2. 结构校验：逐条比对开始时刻、目标、注入 ID、类型、时长、参数，定位首个不一致的注入序号、
   字段名与账本/重放两个取值。
3. 观测校验：逐阶段比对样本计数、成败数、定点成功率与延迟分位数；`null` 语义单独比较，
   非空值按第 3.4 节容差比较，超差即定位到 `注入序号.阶段.字段` 与两侧取值。

## 9. Windows 落地要点

* 挂起/恢复经 `ctypes` 调 `ntdll.NtSuspendProcess/NtResumeProcess`，以
  `OpenProcess(PROCESS_SUSPEND_RESUME)` 取句柄并及时 `CloseHandle`。
* 终止走 `subprocess.Popen.kill()`，Windows 上即 `TerminateProcess`。
* 子进程以 `CREATE_NO_WINDOW` 创建，避免弹出控制台窗口。
* 全部端口绑定 `127.0.0.1` 高位端口；示例使用 18081/18082（服务）与 19081/19082（代理），
  可在声明 JSON 中调整。
* 文件权限 `chmod 0o444` 在 Windows 上转为只读位（`stat.S_IREAD` 语义由 Python 抽象）。

## 10. 已知限制

* 时间轴按确定性顺序串行推进，`budget.max_parallel` 作为运行时并发闸门保留；当前示例
  配置为 1，以保证采样请求序列完全确定。大于 1 时不同目标可并行，但同目标注入不重叠。
* 延迟绝对值受宿主调度影响，不做逐位承诺，只在量化+调度容差内比较。
* Linux 无 `tkinter` 时 GUI 不可启动，使用 `render_headless` 输出同一视图模型的文本报告；
  Windows 官方 Python 自带 tkinter，可直接 `chaoskit gui`。
