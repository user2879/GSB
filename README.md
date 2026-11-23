# 副本修复预算调度诊断工作台（repairbench）

面向 Windows 本地环境的副本修复预算调度诊断工作台：以本地 JSON 声明若干副本组，
在**全局修复并发上限、每节点带宽配额、全局修复字节预算**三重约束下推进批次构造与摘要核对，
以**整数定点量化**核算字节数、权重与预算消耗，给出量化误差界与不可区分判定，
并在预算参数演进与强制终止后保持调度结论可复核。

## 运行环境

- Windows 10/11，Python 3.11+；`python -m pip install numpy`
- Tkinter 随标准库提供（GUI），SQLite 使用标准库 `sqlite3`
- 不使用 Docker 等容器；仅操作 127.0.0.1 回环端口与本地工作目录

## 目录结构

```
main.py                 CLI 入口
repairbench/
  spec.py               副本组/节点/块/预算参数加载与强校验（E_SPEC）
  content.py            确定性块内容生成与修复前摘要登记
  fixedpoint.py         整数定点量化与三段误差界核算（numpy int64 聚合）
  scheduler.py          三重约束批次调度（ThreadPoolExecutor 真实线程池）
  verdict.py            紧迫度排序与不可区分裁决（三类结论）
  evolution.py          版本演进、影响推导与陈旧结论防护
  ledger.py             SQLite 账本与会话快照（单事务提交、整体替换恢复）
  replay.py             恢复后一致性重放校验（E_REPLAY）
  workbench.py          会话编排门面（加载/调度/演进/恢复/注入）
  netprobe.py           127.0.0.1 回环端口监听探测
  gui.py                Tkinter 诊断工作台
samples/                本地确定性样本（组声明、演进声明、端口分配表）
tests/                  43 项单元/集成测试
```

## 关键规则（显式声明）

### 规格校验（E_SPEC）
节点/块标识唯一；host 必须为回环字面量 `127.0.0.1`，端口 1024–65535 且全局不重复；
带宽配额与优先级权重为非负整数；块字节数为正整数；批次段长 ∈ {1KB..1MB 的 2 幂} 且能覆盖块长度；
并发上限与全局预算为非负整数；缩放指数为 0–6。任一非法以 `E_SPEC` 拒绝并指明首个非法条目与字段。

### 确定性块内容与摘要
内容：`chunk_i = SHA256(SHA256(seed|block_id) || u64le(i))` 计数器扩展，截断到块字节数。
摘要：`SHA256("RB1|" || block_id || u64le(bytes) || content)`，修复前逐块登记；
相同声明多次生成逐位一致。

### 定点量化（需求 4）
- scale = 10^e（e ∈ 0..6），定点单位 1 KB，1 量子 = 1024/scale 字节；
- 乘除全部整数完成，**截断方向恒为向零截断**；**溢出饱和**到 int64 两端，除零按符号饱和；
- 误差界三段拆分（量子计）：量化截断（每次量化/除法 < 1 量子）、
  批次切分取整（末段向上取整 0..seg-1 字节）、排队延迟折算（每次折算 < 1 量子）；
- 输出单块字节量化误差、逐节点预算消耗误差界、全局修复量误差上界，并标注主导段。

### 三重约束调度（需求 3）
任一时刻：运行批次总数与单节点运行批次数 ≤ 并发上限；单节点已占用带宽之和 ≤ 节点配额；
累计构造字节数 ≤ 全局预算（派发时预留）。批次占用带宽 = max(1, ⌈批字节/1024⌉) KB/s；
配额/预算不足时排队并记录等待开始/结束的单调时钟毫秒；预算永久不足的批次标记
`DEFERRED_BUDGET`。调度真实投射到 `ThreadPoolExecutor` 工作线程池。

### 不可区分裁决（需求 5）
紧迫度 = `mul(quantize_ratio(失配‰,1000), quantize(权重))`（整数定点合成）。
两候选定点差 ≤ 合成误差界时判 **ERROR_INDISTINGUISHABLE**，按稳定决胜规则
（块标识序、再节点标识序）选定并标注受误差影响；差 > 界为 **EXACT_SEPARABLE**；
失配程度缺失为 **INSUFFICIENT_EVIDENCE**。全序键 (-urgency, block_id, node_id)
保证结论与节点处理顺序无关。

### 演进与陈旧防护（需求 6）
配额/预算/并发/节点集合修改 → 版本号单调递增；沿 节点→块→批次 引用推导受影响
排队批次与被延误块集合；旧版本调度轨迹、紧迫度排序、误差归因整体标记 `stale`
不参与当前展示；当前视图整体替换，稳定列出新增/删除/变更节点及受影响块集合。

### 账本与快照（需求 7）
每完成一个批次，在**同一 SQLite 事务**内写入批次标识、块标识、字节段区间、内容摘要、
带宽占用、等待时长、误差界并推进进度游标；账本携带整数格式版本与内容摘要。
快照与磁盘状态不一致时**整体替换**（删库重建）而非增量合并。
强制终止后重启仅依据已完整提交记录恢复；未提交批次（pending 遗留）标记 `UNKNOWN`
要求重跑，其占用配额先回收。

### 重放校验与异常注入（需求 8）
恢复后按账本预算参数与进度游标重新推进，逐项比较批次结局、块摘要、带宽占用、
紧迫度排序与误差归因（确定段；等待时长与排队延迟折算为墙钟相关量不参与逐项比较），
首个不一致定位到批次/块/字段并以 `E_REPLAY` 报明。命令行注入：
`--inject-corrupt-block`（E_INTEGRITY）、`--inject-oversub`（E_OVERSUB）、
`--inject-snapshot-mismatch`（E_SNAPSHOT），均写入 `workdir/logs/events.jsonl` 结构化日志。

## 使用

```bat
python main.py probe  --spec samples\groups.json --workdir work   :: 回环端口探测
python main.py run    --spec samples\groups.json --workdir work   :: 三重约束调度
python main.py replay --spec samples\groups.json --workdir work   :: 重放校验
python main.py run    --spec samples\groups.json --workdir work --crash-after 2
python main.py replay --spec samples\groups.json --workdir work   :: 崩溃恢复后重放
python main.py evolve --spec samples\groups.json --workdir work --group alpha --new-spec samples\evolve_alpha.json
python main.py report --spec samples\groups.json --workdir work   :: 当前视图（误差界+裁决）
python main.py gui    --spec samples\groups.json --workdir work   :: Tkinter 工作台
python -m unittest discover -s tests -v                            :: 测试
```

## 实际验证结果（本机执行记录）

- `python -m unittest discover -s tests`：**44 项全部通过**（规格校验 14、内容确定性 3、
  定点 4、调度 6、裁决 6、账本/演进/恢复/重放/注入 10、端口探测 1）。
- 三重预算调度（样本 alpha：并发 2、预算 40000B、块共 41624B）：
  11 批次 COMMITTED + 1 批次 DEFERRED_BUDGET，消耗 39576B ≤ 预算；
  运行期峰值并发/单节点并发 ≤ 2、单节点占用带宽 ≤ 配额（仪表化断言）。
- 定点误差界（alpha，scale=10^3）：量化截断 17 量子、批次切分取整 7351 量子（**主导段**）、
  排队延迟折算 11 量子，全局上界 7379 量子 ≈ 7557 字节。
- 不可区分裁决（alpha）：EXACT_SEPARABLE=3、ERROR_INDISTINGUISHABLE=1
  （a-b3 1803 vs a-b4 1800，差 3 ≤ 合成界 8，按块标识序决胜并标注受误差影响）、
  INSUFFICIENT_EVIDENCE=1；输入乱序 5 次结论不变。
- 强制终止（--crash-after 2）：恢复标记 5 个 UNKNOWN 批次并回收配额，
  重放一致（重跑 10 批次，12 条已提交记录逐项比对通过）。
- 异常注入：块内容损坏 → E_INTEGRITY；配额超订 → E_OVERSUB；
  快照摘要不符 → E_SNAPSHOT 且账本整体替换；篡改账本带宽字段 → E_REPLAY
  （定位首个批次/块/字段）。三类注入均写入结构化日志。
- 演进（alpha v1→v2）：新增 a-n4、删除 a-n2、变更 a-n1/a-n3，被延误块 a-b3/a-b4，
  v1 结论整体标记 stale，当前视图整体替换；演进后再运行并重放一致。
