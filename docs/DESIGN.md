# 多环境部署配置一致性仲裁平台 — 技术说明

## 1. 环境适配与前置检查

| 项 | 要求 | 本机实际 | 处理 |
| --- | --- | --- | --- |
| Python | ≥ 3.11 | 3.11.0rc1（`/home/zsl/.feedback/python-base`） | 系统 `python3.11` 缺 `venv/ensurepip`，改用基解释器 `--without-pip` 建虚拟环境并引导 pip |
| SQLite | ≥ 3.40 | 系统库 3.37.2（不达标） | venv 安装 `pysqlite3-binary`，运行时 SQLite 为 3.51.1，见 `arbplatform/db.py` |
| numpy | 1.26 | 1.26.4 | 固定在 `requirements.txt` |
| tkinter | 标准库 | 系统缺 `libtk8.6.so` 且无免密 sudo | 平台核心不依赖 GUI；`envcheck` 显式报告 `tkinter unavailable`，不影响仲裁链路 |
| 端口 | 回环不冲突 | 49101-49103 / 49211-49214 | 声明前用真实 TCP 探测；冲突即 `E_PORT` |

复现环境：`bash scripts/bootstrap.sh`，验证：`bash scripts/run_all.sh`。

## 2. 模块架构

```
arbplatform/
  errors.py       业务错误码 E_SPEC/E_PORT/E_TIMEOUT/E_REPLY/E_REPLAY/E_FIXED/E_STATE
  determinism.py  规范化 JSON（sort_keys）、SHA-256 摘要、单调时钟毫秒
  fixedpoint.py   10^k(k=0..3) 定点量化、零向截断、INT64 饱和、整数乘除与误差界
  spec.py         环境/实例/视图声明加载与定位到首个非法字段的强校验
  samplegen.py    numpy 种子驱动的确定性观测与 NDJSON 事件流
  engine.py       环境内修订生效链 + 三类确定性合并口径
  planner.py      全量/增量工作量显式规模规则与容差决胜
  propagation.py  变更集、视图版本单调递增、失效传播依据
  scheduler.py    全局信号量 + 每环境信号量的真实线程池调度
  service.py      子进程内的回环 TCP 仲裁服务（逐行 NDJSON）
  client.py       父进程子进程监管与真实套接字客户端
  protocol.py     线路协议与回执必填字段校验
  db.py           pysqlite3 适配与 SQLite 版本门禁
  ledger.py       单事务账本、整体替换恢复、未提交标记、E_REPLAY 重放
  platform.py     顶层编排
  cli.py          envcheck/demo/large/recovery 结构化结果入口
```

## 3. 关键确定性规则

* 所有映射经 `canonical()`（`sort_keys=True`、紧凑分隔符、禁止 NaN）后再摘要或持久化；
  列表顺序敏感，所有列表在合并前按显式键（环境标识、实例标识、序号、配置键）排序。
* 环境内生效链：同实例取最高 `seq`，代表观测按 `(-seq, instance_id)` 稳定选取。
* 三类口径：
  * `latest_revision`：全局最高修订序号候选，定点值最小者胜（再按环境标识决胜）。
  * `weighted_majority`：按定点取值分组累计环境权重，`(-weight, q, envs)` 稳定排序。
  * `interval_midpoint`：交集 `[max(lo), min(hi)]` 中点；`lo>hi` 判定口径冲突，不采纳任何环境。
* 同一输入在任意上报顺序、任意批次切分下 `content_digest` 逐字节一致（见
  `tests/test_engine.py::test_order_and_batch_independent_byte_identical`）。

## 4. 定点量化与误差口径

* 缩放因子 `s=10^k`，`k∈{0,1,2,3}`；浮点按最短十进制文本经 `Decimal(str(x))` 精确缩放，
  截断方向统一为向零（`ROUND_DOWN`）。
* 溢出饱和到有符号 64 位 `INT64_MAX/INT64_MIN` 并置 `saturated`；乘除仅用整数，
  除法模拟 C 风格向零截断。
* 单键量化误差上界 = 1 ulp；逐键比较判定误差界 = `2 ulp + tolerance*s`；
  视图级聚合误差上界 = `key_count * 比较误差界`。
* 候选定点差落入合成误差界时标注 `error_influenced`，稳定决胜取较小定点值、再按环境标识。
* 三类成因互斥可叠加：`error_indistinguishable`、`policy_conflict`、
  `rejected_ratio_too_high`（被否占比 > 声明阈值，默认 0.5）。

## 5. 规模自适应路径选择（显式规则，按顺序判定）

1. 变更集为空或受影响键为 0 → 增量 no-op，保留全部既有判定。
2. 观测条目 ≤ 256 → 全量（小输入边界，最简可复核）。
3. 全部键受影响 → 全量（不存在可保留的未失效子集）。
4. 否则估算：
   * 全量工作量 = 环境数 × 键数 × 观测条目数；
   * 增量工作量 = 环境数×变更观测数 + 环境数×受影响键数×max(1,变更观测数)。
   容差带 = `max(1, round(全量工作量 × 0.10))`；工作量差落在带内时稳定决胜为全量并
   标注 `tolerance_influenced=true`；带外取严格较小者。决策输出理由、参数与边界值。

## 6. 失效传播与增量保持

* 视图版本号在每次受影响传播时 `+1`，沿「环境 → 配置键 → 视图引用」推导。
* 传播按视图标识序输出，每步记录 `basis`（哪条变更、经哪个环境/键）。
* 仅重算受影响键，未受影响键直接复用 `KeyVerdict`，比对 `canonical` 记录保证逐字节不变。

## 7. 并发与资源隔离

* 真实 `ThreadPoolExecutor` + 两个 `threading.Semaphore`：
  全局上限 = `global_concurrency`，每环境 = `lease_capacity`。
* 任务同时获得两个信号量才启动，故任意时刻在途总数与单环境在途数均不超限；
  排队的等待起止用 `time.monotonic()` 毫秒记录。不使用 sleep/超时充当隔离。

## 8. TCP 子进程与错误码

* 每环境一个 `python -m arbplatform.service` 子进程，监听 `127.0.0.1:声明端口`，
  逐行 NDJSON 收发；回执含 `env_id/seq_low/seq_high/authority_digest/elapsed_ms`。
* 超时全部来自真实 `socket.settimeout` 与单调就绪截止时间，无人为延迟。
* `E_PORT`：端口占用/绑定失败/连接失败；`E_TIMEOUT`：就绪握手或 IO 超时；
  `E_REPLY`：回执缺必填字段（批量与仲裁回执分别校验）。

## 9. 账本、快照与强制终止恢复

* 每个视图仲裁在单个 SQLite 事务内写权威取值、采纳/被否集合、误差界、归因、版本号；
  账本 `meta` 携带整数 `format_version` 与声明 `spec_digest`。
* 重开快照时格式版本或摘要不一致 → 删除 `views/pending/meta` 整体替换，绝不增量合并。
* 强制终止：`pending` 中未提交视图标记 `status=unknown` 并要求重跑；服务重启仅依据
  已完整提交记录恢复，并向新服务重推当前观测存储。
* 重放逐视图、逐键、逐字段比对，首个不一致即以 `E_REPLAY` 定位到
  `views.<v>.verdicts.<key>.<field>`。

## 10. 验证结果摘要

* 单元 + 集成测试：53 个全部通过（标准库 `unittest`，无需 pytest）。
* `demo`：三环境三口径全量仲裁、单键变更增量重算，未受影响键逐字节保持，账本重放 6 个键判定全匹配。
* `large`：480 条观测、单键变更时选择增量（全量 5760 vs 增量 8，带外严格更小），调度实测在途峰值受双重配额约束。
* `recovery`：`SIGKILL(-9)` 后未提交视图标记 unknown，重启重推 18 条观测后重跑成功；
  摘要不匹配触发整体替换；重放完全一致。
