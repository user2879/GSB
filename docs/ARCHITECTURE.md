# 架构与模块设计

## 1. 总体结构

```
slowquery_diag/
  errors.py          # E_SOURCE/E_FORMAT/E_PORT/E_TIMEOUT/E_ACK/E_REPLAY/E_LEDGER
  models.py          # 来源/候选/归一记录/证据的数据模型与确定性 JSON
  config.py          # JSON 声明加载与强校验（需求 1）
  codec.py           # 多版本二进制帧编解码、字段补齐与归一（需求 2）
  aggregation.py     # NumPy 加权聚合、p50/p95/p99/avg/max 代表值
  arbitration.py     # 显式偏置序的多源证据仲裁（需求 3）
  quantization.py    # 定点量化、饱和、三段误差界（需求 4）
  diagnosis.py       # 不可区分决胜与三类成因归因（需求 5）
  replay_server.py   # 回环 TCP 服务、长度前缀帧、真实超时、回执校验（需求 6）
  scheduler.py       # 子进程托管 + 全局/每来源配额调度（需求 6）
  invalidation.py    # 方案指纹、差异、陈旧标记、整体替换（需求 7）
  ledger.py          # SQLite 账本、版本/摘要、恢复与重放比对（需求 8）
  engine.py          # 编排上述模块的顶层引擎
  cli.py / gui.py    # 命令行入口与可选 Tkinter 界面
scripts/generate_samples.py   # NumPy 确定性样本生成
demos/                        # 三个可重复演示
tests/                        # 30 个 unittest 用例
examples/                     # 声明、帧文件、回放目录
```

## 2. 数据帧格式（多版本、自描述长度）

所有整数均为大端无符号整数：

| 区域 | 字节 | 说明 |
| --- | --- | --- |
| 帧头 | 9+2 | 魔数 `SDQF`、帧头版本 u8、来源类型 u8、格式版本 u8、记录数 u32 |
| 记录 | 变长 | `frame_len` u16、记录版本 u8、字段数 u8、字段 * u64 |

- 首字段固定为 `query_id`；历史版本字段集不同，见 `codec.FIELD_LAYOUTS`。
- `frame_len == 2 + 8*字段数`，解码逐行校验自洽性，首个违规行以 `E_FORMAT` 定位。
- 归一后统一为整数微秒耗时、整数扫描行数；被补齐字段与依据随记录保留。

## 3. 稳定仲裁的全序关系

对每条查询的每个指标，证据按以下全序取最小者：

1. 来源类型偏置：`db_exec_stats` < `slow_log_excerpt` < `app_log_summary`
2. 格式版本降序（键中取负号）
3. 来源标识字典序升序

排序全部显式给出，不依赖 dict/hash 遍历；`canonical_json` 固定 `sort_keys`
与分隔符，因此任意来源上报顺序下输出逐字节一致（测试
`test_byte_identical_under_shuffled_input` 与演示 `demo_error_paths.py` 验证）。

## 4. 定点量化与误差界

缩放因子 `s = 10**k`，`k ∈ {0,1,2,3}`；量化 `q(x)=min(SAT, x//s)`，
截断方向为向零向下取整（输入非负即 floor），溢出饱和到 `2^31-1`。

误差界（原始单位）三段相加：

- 量化截断 `B_quant = 2*(s-1)`：两条比较记录各损失不足一个 tick；
- 代数口径 `B_caliber`：同源同口径为 0；口径不同时取该查询池化记录
  代表值的极差（耗时），或库侧与补齐来源的行数差；
- 采样周期 `B_period`：耗时取 `1000*(最大周期-最小周期)` 微秒，行数为 0。

主导段取原始值最大者；每段同时给出原始值与向下取整的 tick 值。详见
`docs/FIXED_POINT_ERROR_BOUNDS.md`。

## 5. 回放调度与子进程模型

- `LoopbackServiceProcess` 用 `subprocess.Popen([sys.executable, "-m",
  slowquery_diag.replay_server", ...])` 在 127.0.0.1 起独立进程；
  Windows 下带 `CREATE_NEW_PROCESS_GROUP`。
- 真实 TCP + u32 长度前缀 JSON 帧；客户端 `settimeout` 是唯一超时机制，
  仅专用 `stall` 目录条目会真实阻塞以验证 `E_TIMEOUT`，配额从不依赖 sleep。
- 调度器：有界线程池（全局上限）+ 每来源 `Condition` 配额；排队起点/终点、
  开始/结束均用 `time.monotonic_ns()` 毫秒记录。
- 端口占用 `E_PORT`、握手/回执超时 `E_TIMEOUT`、回执缺字段或来源不符
  `E_ACK`。

## 6. 失效传播与账本

- 来源/候选均有 SHA-256 指纹；编辑后 `scheme_version` 单调递增。
- 差异沿 来源→查询→指标→证据→候选回放 引用传播，受影响结论以
  `StaleEntry` 标记陈旧且不进入当前视图；当前视图整体替换并列出
  新增/替换/删除条目。
- SQLite 使用回滚日志模式：硬杀只丢失未提交事务，已提交行在主库文件中。
- 每批次/每次回放单事务写入；`ledger_meta` 保存整数格式版本与全表内容
  摘要，重开时同口径重算，不一致则整体替换并保留只读 `.readonly.bak`。
- 未提交回放（意图行 `status=unknown`）重开后必须重跑；重放结果与账本
  逐字段比对，首个不一致以 `E_REPLAY` 定位到候选与字段。
