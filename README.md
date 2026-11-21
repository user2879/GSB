# 哈希分组聚合连接级水位诊断引擎

面向 Windows/Linux 本地环境的纯 Python（标准库 + NumPy）引擎。核心资源不是行组
分片，而是“聚合连接”：每个分区持有整数连接槽位，受最大并发连接数、每连接可复用
分组键上限、空闲回收毫秒阈值约束，分区之间可声明带归还截止的槽位借用关系。

## 目录结构

```
hashlot/
  errors.py        E_SPEC/E_REPLAY/E_BORROW/E_DEADLINE/E_DIGEST 错误码
  clock.py         单调时钟抽象（真实 monotonic / 可注入 FakeClock）
  fixedpoint.py    int64 定点量化、饱和加减、误差上界
  hashing.py       FNV1A32_V1 显式哈希组合规则与分区/子分区映射
  spec.py          列式表、查询、槽位拓扑加载与 E_SPEC 强校验
  scan.py          列式扫描、过滤、稳定哈希分区与热点再分发规划
  slots.py         连接状态机、借用、空闲回收、水位与等待计时
  aggregation.py   定点局部聚合与确定性合并
  ledger.py        SQLite 账本、批次事务、版本识别与快照整体替换
  replay.py        恢复后逐连接/逐字段重放校验
  engine.py        批次推进、线程池、崩溃注入与恢复水合
  samplegen.py     NumPy 固定种子的本地倾斜样本生成
  cli.py           validate/run/recover/replay 命令行
tests/             40 项 unittest
samples/           本地确定性倾斜样本
docs/              倾斜注入说明与可复现实验步骤
run_lab.py         一键实验脚本
```

## 关键设计

- 稳定哈希：规则 `FNV1A32_V1`，整数按 int64 小端编码，浮点先按会话 scale
  向零截断量化；分区数仅支持 `{1,2,4,...,1024}`，不使用 Python 内建 `hash()`，
  不依赖字典/集合遍历顺序。
- 连接状态机：`established -> active <-> idle -> released`；空闲超时、每连接
  键数达上限、借用截止到期都会真实释放并减少活跃计数。
- 三重约束：全局活跃连接数、每连接分组键上限、出借方“自有占用+借出占用”上限；
  资源不足时排队并记录单调时钟等待区间，输出每连接承载键数与峰值水位。
- 热点再分发：分区行数达整数阈值即按二级哈希再分发，同键恒收敛到唯一子分区，
  子分区间分组键集合不相交；再分发与不再分发聚合结果逐字节一致。
- 定点规则：向零截断量化、int64 饱和加法、均值向零截断；单值误差 `< 1/scale`，
  n 值求和误差上界 `n/scale`，均值再含一次截断误差。
- 账本与快照：每批次单事务提交，携带格式版本与 SHA-256 摘要；版本/摘要不一致
  整体替换并稳定列出被替换表；崩溃后仅依据完整批次恢复，运行中/借用中连接
  标记 unknown 并要求重跑。

## 错误码

| 码 | 触发场景 |
| --- | --- |
| E_SPEC | 规格非法（表/列重复、位图长度、类型相容、借用自环/成环等），含首个违规位置 |
| E_REPLAY | 重放连接字段、借用归属或聚合中间量与账本不一致，定位到连接与字段 |
| E_BORROW | 注入借用关系成环 |
| E_DEADLINE | 注入归还截止早于借用时刻 |
| E_DIGEST | 分片/快照摘要不一致 |

## 快速开始

```bash
pip install numpy
python -m hashlot.samplegen --out samples/skew_sample.json
set PYTHONHASHSEED=42
python -m hashlot.cli --sample samples/skew_sample.json --out-dir out/lab --virtual-clock run
python -m hashlot.cli --sample samples/skew_sample.json --out-dir out/lab --virtual-clock replay
python run_lab.py
PYTHONHASHSEED=42 python -m unittest discover -s tests
```

CLI 支持 `--max-connections/--key-cap/--idle-ms/--partition-count/--batch-size/
--hotspot-threshold/--workers` 覆盖槽位参数，以及三类 `--inject-*` 异常注入。

## 实测验证结果

- `PYTHONHASHSEED` 取 1/42/999 与 4 线程交错下，`report.json` 逐字节一致。
- 默认倾斜样本热点分区 `[4,5,6]` 触发再分发；同键唯一分区/子分区、子分区键集合
  不相交；再分发与不再分发结果摘要相同。
- 峰值活跃连接 4，未超 `global_max_connections=4`；分区 0/1 真实向 6/7 借用连接。
- 批次 5 后强制终止（退出码 7），恢复重放 6 个完整批次后续跑，最终结果与正常
  运行逐字节一致，崩溃时连接标记 unknown。
- E_BORROW/E_DEADLINE/E_DIGEST 注入均按预期报错并写入 `events.jsonl`。
- 全量 `unittest` 40 项全部通过。
