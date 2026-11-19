# aggwin — 聚合与窗口函数流式处理管线 · 技术说明

## 环境

- Python 3.11+（Windows 本地；官方安装包自带 tkinter 与 sqlite3）
- numpy 1.26.4（列式存储与规范参考计算），已用 `pip show numpy` 确认
- 全部样本与测试数据由 `make_samples.py` 本地确定性生成，不访问网络

## 代码结构

| 文件 | 职责 |
| --- | --- |
| `aggwin/schema.py` | JSON 表/查询加载与强校验（E_FORMAT，输出字段路径） |
| `aggwin/engine.py` | 分批流式推进、幂等行应用、E_BATCH/E_ORDER 检测 |
| `aggwin/state.py` | 分组聚合定点状态、窗口环形缓冲、重排缓冲、状态摘要 |
| `aggwin/fixedpoint.py` | 定点缩放与显式截断方向（floor/ceil/toward_zero） |
| `aggwin/persistence.py` | 状态日志段（二进制 v1/v2）与快照清单（JSON v1/v2）编解码 |
| `aggwin/recovery.py` | 日志回放 / 快照重建两条恢复路径与 E_RECOVER 比较 |
| `aggwin/verify.py` | 规范参考结果、E_WINDOW/E_DIVERGE 定位 |
| `aggwin/catalog.py` | SQLite 目录：规范化标识、参数版本单调递增、快照登记/陈旧标记 |
| `aggwin/faults.py` | 四类异常注入（仅改输入文件或参数） |
| `aggwin/cli.py` | run / recover-check / diverge-check / verify 子命令 |
| `verify_tool.py` | 独立验证工具（只读本地文件与 SQLite） |
| `gui.py` | Tkinter 界面（查询/聚合/窗口/游标/快照五个页签） |
| `demo.py` | 端到端验收演示；`run_tests.py` 28 项测试 |

## 关键设计

### 规范化标识与参数版本
- 查询规范化标识 `q-<sha256前16位>`：对查询定义的规范 JSON（键排序、紧凑分隔符）取摘要，跨进程稳定。
- 参数集指纹 = 全部查询定义 + scale + truncation 的摘要；SQLite `query_param_sets`
  表把指纹映射为单调递增整数版本。任何查询/缩放/窗口参数修改都会产生新版本，
  旧版本快照由 `mark_stale_before` 标记为陈旧（stale=1），重新打开时整体替换
  （替换字段集合：cursors / applied / aggs / windows），旧快照文件只读保留。

### 定点聚合
- 状态全部为缩放整数 `scaled = value * scale`；float 经 `Decimal(repr(v))` 精确转换后
  按会话截断方向量化。均值 = trunc(sum/count)，量化误差上界 `1/scale`
  （样本 scale=1000 → 上界 0.001），在结果中以 `mean_error_bound` 字段给出。
- 每分组记录 count/sum/min/max/rows_processed/group_version（= 参数版本号）。
- 幂等：引擎按表维护已应用行区间，`apply_rows` 自动去重；同批次重复推进为无操作。

### 窗口环形缓冲
- 每分区：history（按 (ts, seq) 排序的全部到达行）、pending（重排缓冲）、
  ring（容量 = max_rows 的环形缓冲，当前窗口内容）、results（行号 → 窗口聚合）。
- 水位线 = max_ts_seen − max_lateness；ts + following ≤ 水位线时按 (ts, seq) 序发射。
- 乱序行（ts < 水位线）按显式策略处理：`recompute` 插入 history 并局部重算受影响行；
  `drop` 丢弃并计数。finalize 做全量对账，保证最终结果与环形缓冲内容在任意批大小、
  任意到达顺序下逐位一致（recompute 策略）。
- 迟到比例超过 `max_out_of_order_ratio` → E_ORDER（定位查询与分区键）。

### 持久化格式
- 状态日志（段文件 `statelog/segment-NNNNNN.log`），记录布局小端：
  - v1: magic `AWL1` | fmt u32 | seq u64 | type u8 | len u32 | payload | crc32 | sha256
  - v2: magic `AWL2` | fmt u32 | seq u64 | checkpoint_seq u64 | type u8 | len u32 |
        payload | crc32 | parent_digest 32B | sha256
  - 单条记录一次 `O_APPEND` 写入 + fsync，保证原子可见；解码校验 magic、版本、
    长度自洽、crc、内容摘要，任一失败 → E_LOG（给出段文件与偏移）。
- 快照清单（JSON）：v1 平铺 `parent_snapshot_id`，v2 用 `parent{snapshot_id,digest}`；
  两者均携带 format_version、query_param_version、内容摘要，解码后归一为同一内存
  结构，跨版本归一不改变输出（测试 `test_cross_version_same_output` 验证摘要一致）。

### 恢复与一致性
- 日志回放：从最近检查点装载状态，再按单调日志序号应用后续批记录（含 cursor_after），
  行级幂等去重。
- 快照重建：按清单整体装载，不做增量合并。
- 两条路径的状态逐字节比较（E_RECOVER 定位首个不一致的分组键/分区键/字段）。
- 恢复续跑 vs 不中断推进：聚合结果、窗口结果、状态摘要三者逐位比较，
  差异定位到分组键/分区键/时间戳/字段（E_DIVERGE）。

## 异常注入（仅改输入文件或参数）

| 注入 | 业务码 | 机制 |
| --- | --- | --- |
| `--inject snapshot-digest` | E_SNAPSHOT | 篡改清单 digest 字段 |
| `--inject log-truncate` | E_LOG | 截断最新段文件尾部 7 字节 |
| `--inject batch-mid-window` | E_BATCH | 开启窗口对齐严格检查 + 批大小 3 |
| `--inject out-of-order` | E_ORDER | 到达乱序窗口 25 + 比例上限 0.05 |

全部异常写入 `logs/structured.jsonl` 结构化日志（JSONL，含 code 与上下文）。

## 实际验证结果（本地样本，60 行 events 表，2 条查询）

- `python3 run_tests.py` → **Ran 28 tests, OK**
- `python3 demo.py` → 全部验收步骤通过：
  - 日志回放与快照重建状态摘要一致：
    `926dc1c9…4580f4`（log_replay_digest == snapshot_digest）
  - 快照恢复续跑（batch=5）与不中断推进（batch=7）状态摘要逐位一致
  - 批大小 {1,3,11} × 乱序窗口 {0,4,6} 组合下聚合与窗口结果完全一致
  - 四类异常分别返回 E_SNAPSHOT / E_LOG / E_BATCH / E_ORDER 并落结构化日志
  - 独立验证工具：恢复路径一致、游标自洽（无重复累加/遗漏行）、结果与状态一致
  - 参数演进（scale 1000→100）：参数版本 1→2，旧快照标记 stale，
    重新打开时整体替换并列出被替换字段集合
  - 日志 v1 + 快照 v1 与 v2 混合运行，跨版本归一后输出摘要一致

## 使用

```bash
python make_samples.py                 # 生成本地样本
python -m aggwin.cli run --tables sample/tables.json --queries sample/queries.json \
    --workdir runs/full                # 不中断运行（快照+状态日志）
python -m aggwin.cli run ... --resume-from runs/full/snapshots/manifest-v2-mid.json
python -m aggwin.cli recover-check ... # 回放 vs 快照
python verify_tool.py --snapshot ... --statelog ... --catalog ... --results ...
python gui.py --run-dir runs/full      # Tkinter 界面（Windows 本地）
```
