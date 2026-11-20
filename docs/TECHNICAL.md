# configdist 技术说明

面向 Windows 本地环境的跨组件配置分发平台（模拟）。纯标准库 + NumPy，
不使用 Docker 或任何外部服务；全部样本由仓库内脚本离线生成。

## 架构

```
configdist/
  manifest.py      清单 v1/v2/v3 解码、强校验、记录级升级、v3 编码
  normalize.py     目录归一：检查点续做、原子重写、只读备份
  digest.py        派生关系求值与期望聚合摘要（平台/节点共用）
  canon.py         规范化 JSON、整数定点运算、聚合口径
  samples.py       确定性样本生成（sha256 派生种子 + NumPy PCG64）
  registry.py      节点注册表校验
  node.py          节点子进程：回环 TCP + 逐行 JSON，状态持久化
  orchestrator.py  子进程编排、握手、回执强校验（E_PORT/E_TIMEOUT/E_PROTO）
  scheduler.py     优先级并行调度：全局上限 + 每节点配额 + 超时重试
  ledger.py        SQLite 账本：批次/切片/回执/配额/进度游标
  platform.py      门面：下发、收敛比对、预览缓存、权威重算、陈旧防护
  latency.py       单调时钟延迟观测（均值/P95/最大值定点表示）
  cli.py / gui.py  命令行入口 / Tkinter 界面入口
```

## 关键设计

### 定点与逐字节一致（要求 4、6）
配置数值与样本数据均为 int64 milli 定点（1000=1.0），乘除用向零截断的
整数运算（`canon.tdiv/fp_mul/fp_div`）。摘要为规范化 JSON（键排序、紧凑
分隔符），平台侧 `platform.py` 与节点侧 `node.py` 共用 `digest.py`
计算，比较时对摘要字符串/digest 做逐字节相等判定。

### 格式强校验（要求 1）
`manifest.decode_manifest` 检测 `format` 字段分派 v1/v2/v3；未知版本、
必需字段缺失、重复键（含 JSON 对象内重复键，经 `object_pairs_hook`
检测）、不支持的键类型（float/null/list/dict）、未知字段、派生引用
不存在或非数值键，均以 `E_FORMAT` 整体拒绝，异常携带 `path`
（如 `entries[3].value`）与 `source_version`。

### 归一与中断续做（要求 2）
`normalize.py` 先整体强校验源清单（失败零副作用），再逐条记录升级，
每条完成后以 tmp+`os.replace` 原子更新 `<file>.upgchk.json`（含已升级
记录与 `next_index`）。全部完成后原子重写清单为 v3，旧内容写入
`<file>.<ver>.bak` 并 `chmod 0444`。强杀后重启只依据检查点续做，
`resumed_from` 之前的记录不重复处理。报告稳定列出源/目标版本。

### 节点编排与协议（要求 3）
节点为独立子进程（`python -m configdist.node`），绑定注册表声明的唯一
回环端口。协议为逐行 JSON：`hello/hello_ack`、`apply/receipt`、
`shutdown`。回执按 s1（仅已应用清单版本+digest+耗时）/s2（增加键级
版本、聚合口径与完整摘要）校验。端口占用退出码 3 → `E_PORT`；握手
超截止时间 → `E_TIMEOUT`；回执缺字段/类型错误 → `E_PROTO`。节点每次
apply 后原子持久化状态，强杀重启后 `hello_ack` 携带恢复状态继续应答。

### 优先级调度（要求 5）
`scheduler.Scheduler` 按 `(-priority, key, node)` 稳定排序，工作线程数
= 全局并发上限；领取任务前检查每节点配额。下发带真实套接字超时，
仅 `E_TIMEOUT` 按 `strategy.max_retries` 重试。`snapshot()` 提供线程
安全的进度与配额占用视图，界面经 `after()` 轮询保持可交互。

### 快速视图与权威重算（要求 6）
界面只读预览缓存（`ui_query`）；`recompute_authoritative` 重放当前版本
全部已提交批次（`replay_batch` 逐条重算期望摘要并与账本比对，差异报
`E_REPLAY`），再按最新批次重建结论，与快速结论逐字节比较，不一致报
`E_PREVIEW`（键与节点）。两类操作均以 `time.monotonic_ns()` 记录整数
毫秒差，统计均值/P95/最大值（milli-ms 定点）与量化误差上界
（±0.5ms/次），P95 超预算记 `E_LATENCY`。

### 陈旧防护与整体替换（要求 7）
编辑清单经 `edit_manifest` 强校验后版本号自动单调递增（账本 meta 持久化
`last_manifest_version`），旧版本批次标记 `superseded`（此后只读），
预览缓存全部标记 `stale` 且不参与当前展示；下一次下发后视图整体替换，
`replacement_report` 稳定列出变化键、受影响节点与旧新摘要。

### 强杀恢复（要求 8）
`begin_batch` 单事务写入 running 批次与全部切片；`commit_batch` 单事务
写入回执、配额、进度游标并置 committed。重启 `recover()` 把 running
批次标记 `unknown` 并要求重跑，不生成退出报告；恢复后按账本重放并
重算派生，首个不一致以 `E_REPLAY` 报明键/节点/字段与依据批次。

## Windows 运行

```
python -m configdist.cli gen-samples --workdir demo_ws
python -m configdist.cli normalize   --workdir demo_ws
python -m configdist.cli run         --workdir demo_ws --manifest app_v3.json
python -m configdist.cli recompute   --workdir demo_ws
python -m configdist.cli recover     --workdir demo_ws
python -m configdist.cli demo        --workdir demo_ws
python -m configdist.gui             --workdir demo_ws
python -m unittest discover -s tests -v
```

注：本仓库验证在 Linux(WSL) 环境完成；代码仅使用跨平台 API
（`os.replace`/socket/subprocess/SQLite），Windows 命令相同。
Tkinter 缺失时 GUI 入口打印提示并以退出码 2 结束，核心功能不受影响。
