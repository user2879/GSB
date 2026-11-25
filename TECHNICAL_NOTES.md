# 灰度路由回执系统 — 技术说明

本地确定性实现，无容器、无外部网络目标。所有后端仅绑定 `127.0.0.1`，端口取自
`config/backends.json`（范围 1024–65535），启动前做占用探测，被占用时改配置即可。

## 环境

- Python 3.11+；仅第三方依赖 `numpy`；`sqlite3`、`tkinter` 走标准库。
- 本机 `tkinter` 的原生 Tcl/Tk 运行库缺失（`libtk8.6.so` 不存在）。界面因此拆为
  纯 Python 无头视图模型 `grayroute/viewmodel.py` 与可选 Tk 外壳
  `grayroute/ui_tk.py`；后者在无 Tk 时返回降级报告而非抛错，核心一致性不依赖显示层。

## 目录与模块

```
config/                 本地 JSON 声明（backends/rules/merge_rules）
data/                   NDJSON 请求流、回执样本、SQLite 账本、快照、报告
grayroute/
  constants.py          受支持枚举、回环常量、桶数、权重范围
  errors.py             E_SPEC/E_FIELD/E_VERSION/E_REPLAY/E_SNAPSHOT
  canonical.py          显式稳定序列化（排序键、固定分隔符）与 SHA-256 摘要
  spec.py               JSON/NDJSON 加载与强校验（需求1）
  receipt.py            r1/r2/r3 编解码、归一、补齐依据（需求2）
  sharding.py           稳定哈希分桶与逐请求命中归属（需求4）
  inference.py          确定性推理与输出摘要（子进程与网关共用同一纯函数）
  batching.py           确定性批计划与批标识（需求3）、E_VERSION 守卫（需求5）
  backend_server.py     子进程托管的回环推理后端（行分隔 JSON 协议）
  backend_proc.py       端口探测、子进程生命周期、回环客户端
  merger.py             跨版本依序归并与失效传播（需求6）
  scheduler.py          全局/每工作器配额的并行调度与单调时钟等待（需求7）
  ledger.py             SQLite 模式、线程局部连接、单事务原子提交（需求7）
  snapshot.py           版本识别快照与整体替换恢复（需求8）
  recovery.py           强制终止未知批次标注与 E_REPLAY 重放（需求8）
  viewmodel.py/ui_tk.py 快速结论 vs 账本权威重算的逐字节一致性
  engine.py/cli.py      编排与命令行入口
tests/                  37 个 unittest 用例
scripts_*.py            样本与验收生成器；run_demo.sh 一键运行
```

## 需求到实现的对应

1. 强校验：`spec.py`。后端/规则标识唯一、端口必须是字面量 `127.0.0.1` 且全局不重复、
   维度为正整数、量化与参数格式受枚举约束、权重 0–10000、规则修订号严格递增、到达序号
   不重复、向量长度等于 `model_dim`。首个非法对象的标识与字段随 `E_SPEC` 返回。
2. 三版回执：`receipt.py`。归一记录带来源 `format_version`、`filled_fields` 与逐项
   `fill_basis`；校验非负耗时、批内名次不重复不越界、重复投递引用存在、参数摘要长度，
   违规以 `E_FIELD` 定位首字段；归一内部表示走 `canonical()`，重复解码逐字节一致。
3. 确定性批计划：`batching.py`。按工作器+参数版本分组、到达序号排序、修订批大小切片；
   批标识 = `bat-` + 对 `{worker_id,batch_index,arrival_seqs}` 稳定序列化的 SHA-256
   前缀，与到达/执行顺序无关；每批记录修订号、参数版本序号与请求集合。
4. 稳定哈希：`sharding.py`。`sha256("rule=..|client=..") mod 10000` 得 0–9999 桶，
   与权重比较决定 stable/gray；工作器选择用独立哈希且先排序候选，去除遍历顺序影响；
   命中同时绑定批、工作器与当时参数版本序号。
5. 热更新隔离：分组键含参数版本序号，新旧版本进入不同批；批级 `check_batch_versions`
   在混版时抛 `E_VERSION`，附带冲突批与两版本摘要；后端按冻结的 `param_summary` 推理。
6. 归并与失效：`merger.py`。按参数版本降序、格式版本降序、归并序号升序选采纳回执，
   其余保留并记录否决理由；`propagate_invalidation` 沿
   工作器参数版本→批→请求 推导，按工作器、版本、批、请求稳定输出；未失效批的
   `content_summary` 不受影响。
7. 调度与原子提交：`scheduler.py` 用信号量强制全局上限与每工作器配额，记录
   `monotonic_ns()` 换算的等待起止毫秒；`ledger.py` 用线程局部连接 + 提交锁，
   批归属/采纳回执/否决/失效轨迹/进度游标在 `BEGIN IMMEDIATE…COMMIT` 单事务写入，
   提交前对其他读取者不可见；账本带整数 `format_version` 与内容摘要。
8. 快照与恢复：`snapshot.py` 持久化规则/工作器集合/参数版本、账本摘要与游标并自带摘要；
   版本或摘要不一致即整体替换。`recovery.py` 只认已提交记录，未提交批标 `unknown`
   并要求重跑，不生成退出报告；重放逐字段比较，首个不一致以 `E_REPLAY` 定位
   批标识、请求标识与字段。

## 界面一致性

`viewmodel.consistency_check()` 同时计算内存快速结论与重读 SQLite 的权威结论，比较两者
的 canonical 字节；实测 `byte_identical=true`、耗时约 1ms，处于 5000ms 交互预算内。
