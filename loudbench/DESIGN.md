# 响度与动态范围并行测量工作台 — 技术说明

## 1. 总体结构

```
loudbench/
  errors.py      业务错误码与可复算异常（带 location/evidence）
  wavio.py       确定性 48kHz/24bit/6 声道 WAV 读写
  synth.py       固定种子 5.1 合成素材与分段真值标签
  kweight.py     BS.1770-4 K 加权（冲激响应+分块 FFT 卷积）与 4x 过采样真峰值
  meter.py       400ms/100ms 积分块、两级门控、LRA、整段参考实现
  tasks.py       时间窗切分、块末端归属、确定性任务清单
  bufferpool.py  规范 LRU 计划、驻留/回收账本、运行时共享缓冲池
  scheduler.py   线程池调度、内存/耗时预算、固定优先级降级、时间轴归并
  report.py      归并报告、参考比对、确定性 JSON 与 SHA256
  demo.py        一键固定序列演示与异常注入
  app.py         PyQt5 响度曲线、逐窗表格与预算/驻留展示
```

## 2. 确定性保证

- 合成：所有随机量来自 `numpy.random.default_rng(seed)`，无时间/环境随机量；
  WAV 仅写标准 PCM 头与量化载荷，同一阵列两次写入逐字节一致。
- 切分：任务键 `T0000...`、窗口边界、块锚点全部由样本数与固定窗口参数决定，
  清单以 `sort_keys` JSON 序列化。
- 测量：全部使用 float64；积分块锚定在整段起点的全局 100ms 网格上，块集合与
  窗口切分无关，因此线程数不进入任何数学运算。
- 调度：驻留决策先做一次“时间轴顺序”的规范 LRU 模拟（canonical plan），运行时
  缓冲池只执行该计划；并发只改变执行时机，不改变 HIT/STORE/EVICT/RECOMPUTE 账本。
- 报告：JSON 固定键序、固定缩进、无时间戳、无临时路径；另输出 SHA256 清单。

## 3. 任务边界的滤波与积分窗延续

- K 加权是 IIR，工程上用其截断冲激响应（0.5s，24k 点）做 FFT 卷积。每个任务携带
  `WARMUP=24000` 个前置真实样本重建滤波状态，warm-up 输出不进入统计，与连续处理
  的偏差远低于 0.1 LU 声明容差（测试中与整段参考一致）。
- 400ms 积分块采用“块末端归属”：块 `(a, a+19200]` 归到包含其末端的任务；跨窗口
  边界的块通过该任务的前置/尾部 warm-up 上下文读取。归属在全部任务间唯一，
  `tasks._validate_partition` 校验无重复、无遗漏。
- 真峰值：4x 零插补 + 窗化 sinc FIR，任务只在自身 `[start,end)` 输出窗口取最大值，
  warm-up 仅用于 FIR 边界，因此相邻任务不会重复认领峰值。

## 4. 预算模型与固定优先级降级

中间缓冲按 10ms 网格（480 样本/块）切分，键形如 `raw:00004800:00005280`，相邻任务
共享同一批键；重采样在 48kHz 直通下为恒等映射，仍以 `rsm:*` 独立记账。

降级固定优先级（可复现，无运行时启发式）：

1. 内存轴：预算不足先 LRU 淘汰（EVICT），再次需要记 RECOMPUTE；单缓冲超总预算或
   淘汰后仍无法预留时记 `E_BUDGET_DENIED`，走 UNCACHED 重算，任务不丢失。
2. 时间轴：单任务确定性估算耗时超预算时，按超出倍数对并行度减半，最低降到 1（串行
   回退），并在报告 `degradation` 中记录原因与数值。

账本事件：`STORE`（首次驻留）、`HIT`（命中复用）、`EVICT`（LRU 回收）、
`RECOMPUTE`（曾驻留被回收后重算）、`DENY`（预算拒绝、不驻留重算）。被淘汰的键
再次访问只可能是 RECOMPUTE/UNCACHED，绝不可能记为 HIT。

## 5. 错误码

| 码 | 触发 | 可复算定位 |
| --- | --- | --- |
| E_AUDIO_FORMAT | 声道数/位深/采样率非法 | 证据含实际 channels/width/rate |
| E_WINDOW_INVALID | 窗口非正、重叠覆盖窗口、短于 400ms | 证据含窗口参数 |
| E_BLOCK_BOUNDS | 重叠超 warm-up、分区不连续、块重复/遗漏 | 证据含任务键与锚点 |
| E_BUDGET_DENIED | 缓冲超预算或 LRU 后仍无法预留 | 证据含任务键、缓冲键、预算字节 |
| E_THREAD_RANGE | 线程数 <1 或 >64 | 证据含请求线程数与上下限 |
| E_MERGE_ORDER | 归并键序列/时间序错乱（含注入故障） | 证据含 got/expected 键序列 |
| E_PEAK_OVER | 真峰值 > 0 dBTP | 证据含任务键、声道、dBTP、过采样索引 |

## 6. 容差声明

- 整段相对门控响度、绝对门控响度：与无预算/整段参考偏差 ≤ 0.1 LU。
- 真峰值：≤ 0.1 dB。
- 积分块数量：必须精确相等（容差 0）。

## 7. 运行方式

```bash
pip install -r requirements.txt
python run_demo.py --workspace workspace/demo --verify-twice   # 一键演示+两次比对
python run_tests.py                                            # 单元/验收测试
python run_gui.py workspace/demo/runs/S1_unrestricted_4w.json  # 图形界面
```

Windows 下可直接使用 `run_demo.bat`、`run_gui.bat`。无显示环境时设置
`QT_QPA_PLATFORM=offscreen` 即可对界面做离屏自检。

## 8. 已验证结论（本机记录）

- 10 个 unittest 全部通过；并行（1/2/4/8 线程）与串行的逐任务结果逐字段一致。
- 紧张预算触发 EVICT+RECOMPUTE，极限预算触发 E_BUDGET_DENIED+UNCACHED，窗口无缺失，
  整段响度/真峰值均在声明容差内并 PASS。
- 两次完整运行的 WAV、标签、任务清单、各运行报告与哈希清单逐字节一致。
