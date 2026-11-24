交接文档只写下了需求条目，仓库里还没有任何代码，需要在 Windows 本地环境从零构建一个列式页存储的崩溃恢复诊断组件，交付可集成的 Python 包、Tkinter 诊断界面与可重复演示脚本。用户在本地 JSON 文件中声明若干列式页（页标识、页内行数、列定义与空值位图）与一段确定性 WAL 日志条目（日志序号、事务标识、页标识、行区间、新值生成种子、提交标记与持久化标记），并声明恢复预算（全局并行回放槽位上限、每事务回放配额、单页回放字节预算与检查点间隔条目数）。系统须以本地 SQLite 维护页版本表与日志元数据，以本地文件持久化页内容，按日志顺序推进页版本并记录每次回放的页内容摘要；恢复过程须能在检查点处继续，并支持强制终止后的恢复重放校验。核心冲突在于检查点粒度、账本原子提交粒度与并行槽位分配共同决定强制终止后的重做窗口：检查点过密拖慢回放却缩小重做范围，账本每批提交保证确定性却放大提交开销，而崩溃点之后未提交事务的页版本不可见又与检查点记录的应用位置存在重叠，必须靠已提交账本与检查点摘要交叉校验重建可见版本集合。

任务环境与准备：
安装 Python 3.11 或以上版本，并确保 python、pip 在 PATH 中可用；本项目不依赖 GPU 或外部服务。
执行 pip install numpy==1.26.4 安装 NumPy；SQLite 与 Tkinter 使用 Python 标准库自带模块，无需额外安装。
在本地工作区准备一个空目录，创建 samples 子目录用于放置页声明与 WAL 日志 JSON 样本，创建 state 子目录用于 SQLite 账本与页内容文件。
确认本地回环端口未被占用（题目不要求真实回环通信，但若自行扩展需使用 127.0.0.1 高位端口）。
运行前以 python -c "import sqlite3, tkinter, numpy" 检查依赖可用。

业务要求：
1. 实现列式页声明与 WAL 日志条目的本地 JSON 加载及强校验：校验页标识与日志序号唯一、页内行数为正整数、行区间不越出所属页、事务标识在已提交日志中成组出现、提交标记与持久化标记组合合法；任一不合法须以 E_WAL 业务码拒绝并指明首个非法日志序号与字段。
2. 实现按日志顺序的确定性页版本推进与页内容摘要登记：以页标识、版本号与日志序号按显式规则生成页字节内容，每次应用后写入 SQLite 页版本表并登记摘要；相同日志在任意批次切分与任意回放槽位分配下必须得到逐位一致的页内容与摘要。
3. 实现受全局并行回放槽位上限与每事务回放配额双重约束的并行回放调度：任一时刻运行回放任务数与单事务运行任务数不得超限，配额耗尽时任务排队并记录等待开始与结束的单调时钟毫秒；调度须真实投射到工作线程池且不得绕过配额。
4. 实现检查点写入与崩溃点判定：每完成声明条目数即写检查点（含已应用日志序号、页版本摘要与内容摘要），崩溃点依据提交标记与持久化标记判定，崩溃点之后未提交事务的页版本须标记为不可见且不参与恢复后页视图。
5. 实现恢复账本的原子落盘与强制终止后的恢复：每完成一个回放批次须在同一 SQLite 事务中写入批次内页版本、摘要、任务结局、预算消耗与进度游标；进程被强制终止后重启只能依据已完整提交的记录继续，未提交批次标记为未知并要求重跑。
6. 实现恢复后的逐批重放校验与差异定位：从最近检查点继续按账本记录的轮次重新回放剩余日志，与不中断一次回放同一区间在页版本号、页内容摘要、已提交事务集合与不可见版本集合上逐项比较，差异须定位到首个不一致的页标识与字段并以 E_REPLAY 报告。
7. 实现回放预算参数或页集合演进后的整体替换与陈旧结论防护：预算参数或页集合被修改后版本号单调递增，旧版本下已产生的页版本视图、检查点与统计须整体标记为陈旧且不得参与当前展示；当前视图须整体替换而非增量合并。
8. 提供 Tkinter 诊断界面与显式异常注入：界面展示页列表与版本号、日志区间、回放调度时间线与预算占用、检查点集合、崩溃点与不可见版本标记、账本版本与快照列表，渲染顺序稳定；支持以命令行参数分别注入日志提交标记冲突、页行区间越界、账本内容摘要不符三类异常，分别以 E_COMMIT、E_RANGE、E_SNAPSHOT 报告。

执行约束：不使用 Docker 或其他容器环境；使用本地确定性样本验证，不访问未授权目标。按上述任务意图交付相应代码、测试或技术说明，记录实际验证结果。

验收：
使用本地样本验证页版本推进、并行回放预算、检查点续做与强制终止后重放校验，并检查异常注入的业务码。

---

# 交付实现

本仓库已包含可集成 Python 包 `src/colpage_recovery/`、Tkinter 界面、确定性样本、测试和可重复演示。详细设计见 `docs/DESIGN.md`，实际验证记录见 `docs/VERIFICATION.md`。

## Windows 快速开始

在仓库根目录打开 PowerShell：

```powershell
python --version
python -m pip install numpy==1.26.4
python -c "import sqlite3, tkinter, numpy; print('deps ok', numpy.__version__)"

# 无需安装包，脚本会自动把 src 加入 PYTHONPATH
python .\scripts\demo.py
python .\scripts\run_tests.py

# 打开 Tkinter 诊断界面
$env:PYTHONPATH = "$PWD\src"
python -m colpage_recovery.cli gui --state-dir state\demo --report artifacts\demo_report.json
```

也可以安装为包后使用 `colpage-diag`：

```powershell
python -m pip install -e .
colpage-diag demo
colpage-diag gui --state-dir state\demo --report artifacts\demo_report.json
```

或直接运行 `scripts\Run-Demo.ps1`。

## 常用命令

```bash
# 校验样本
PYTHONPATH=src python -m colpage_recovery.cli validate --samples samples

# 一键演示：确定性对照 -> 子进程 os._exit(137) -> unknown 批次 -> 恢复校验 -> G2 演进
python scripts/demo.py

# 测试
python scripts/run_tests.py

# 三类显式异常注入
PYTHONPATH=src python -m colpage_recovery.cli diagnose --inject commit-conflict
PYTHONPATH=src python -m colpage_recovery.cli diagnose --inject range-overflow
PYTHONPATH=src python -m colpage_recovery.cli diagnose --inject snapshot-mismatch

# 查看当前 SQLite 视图
PYTHONPATH=src python -m colpage_recovery.cli view --state-dir state/demo
```

命令行业务码：

- `E_WAL`：一般 JSON/WAL 强校验失败，含首个 `lsn/field`
- `E_COMMIT`：同事务提交标记冲突
- `E_RANGE`：行区间越界
- `E_SNAPSHOT`：账本、检查点摘要与页文件/重放结果不符
- `E_REPLAY`：恢复重放与不中断参考回放不一致
- `E_BUDGET`：单页字节预算不足
- `E_GUI`：当前 Python 无法加载 Tk/Tcl（Windows 官方 Python 一般不会出现）

演示报告默认写入 `artifacts/demo_report.json`，SQLite 与页文件默认写入 `state/demo/`。
