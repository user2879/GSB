# 技术说明

## 1. 块格式（colval/formats.py）
块以规范化 JSON（键排序、紧凑分隔符）持久化于 SQLite `blocks` 表。
- **b1**：`format/block_id/row_count/columns[{name,values}]`，仅列标识与行数；
  不允许 null。升级时推断类型（全整数值→int64，含浮点→float64），空值位图
  全有效，量化指数与批参数取默认值（3 / 1024）。
- **b2**：增加 `type` 与 `null_bitmap`（hex，长度必须等于 ceil(row_count/8)）。
- **b3**（当前版本）：再增加 `quant_scale_exp` 与 `batch_max_rows`。

强校验在解码时执行，任何未知版本、必需字段缺失、位图长度不符、类型不受
支持、值数量与行数不符、b1 出现 null，均以 `E_FORMAT` 拒绝并携带
`block_id` 与首个非法 `field`（如 `columns[0].type`）。

内容摘要：`content_bytes()` 把块头 + 每列（名、类型标签、位图、小端
int64/float64 值序列）序列化为规范字节流后取 SHA-256，作为逐字节一致性
的判定依据。

## 2. 升级与断点续做（colval/store.py）
打开数据目录时按 block_id 排序逐块处理：
1. 检查点 `checkpoint.json` 已记录的块一律跳过（强制终止后重启只依据检查点
   继续，已升级块绝不重复处理）；
2. 解码校验旧块 → 内存中升级为 b3；
3. **单个 SQLite 事务**内完成：旧载荷写入 `blocks_backup`（INSERT OR IGNORE，
   只增不改不删，即只读备份）+ 替换 `blocks` 行；
4. 随后以临时文件 + `os.replace` 原子更新检查点。
若在 3、4 之间被强杀，重启后至多重做一块，且备份仍保留首份原始载荷。

## 3. 逐字节一致性校验（colval/verify.py）
逐列比较列名、类型、空值位图与值（float 按 IEEE754 小端字节比较），再比较
内容摘要；首个不一致以 `E_DIVERGE` 抛出并定位 `block_id`、`column`、`row`。

## 4. 表达式与计划（colval/expr.py, colval/plan.py）
递归下降解析器支持 `+ - * / %`、比较、`and/or/not`（及 `&&/||/!`）、
`cond ? a : b` 与 `if(c,a,b)` 三元。规范化为完全括号化的前缀 s 表达式，
因此仅空白/括号不同的同语义表达式得到相同 SHA-256 规范化摘要。
批切分产生首尾相接、不重叠、不遗漏的 `[start,end)` 区间；显式声明的批
区间若超出总行数、重叠、留空或倒序，均以 `E_PLAN` 拒绝。

## 5. 定点量化与误差传播（colval/quant.py）
- 会话缩放因子 `S = 10**scale_exp`；`quantize(x) = trunc0(x*S)`，截断方向
  **向零**；溢出**饱和**到 int64 两端；乘 `(qa*qb) trunc-div S`，除
  `(qa*S) trunc-div qb`，所有截断除均向零。
- 单值量化误差上界 `1/S`；误差界按算子传播：加减 `ea+eb`，乘
  `|b|ea+|a|eb+ea·eb+1/S`，除给出保守闭式（分母下界 ≤0 时为 +∞）；逐批聚合
  误差界 = 行数 × 单行界。
- 候选表达式排序：定点结果差 ≤ 合成误差界（量化域）时按稳定决胜规则
  （规范化摘要 → 规范化文本）排序并标注 `error_affected=True`。

## 6. 双约束调度（colval/scheduler.py）
条件变量调度器：任一时刻运行批内存单元之和 ≤ 全局预算，且每任务运行批数
≤ 配额；否则排队。每批记录入队/开始/结束的单调时钟毫秒、排队时长、内存
单元数与峰值字节（单元 = 1 KiB，`ceil(rows×cols×8/1024)`，至少 1）。

## 7. 向量化求值与确定性合流（colval/executor.py）
批内逐算子对整列（numpy object 数组承载 Python 精确整数）求值。空值传播：
算术/比较/三元任一空操作数得空；逻辑采用三值规则（`false AND null=false`，
`true OR null=true`，`NOT null=null`）。除/模零：该行结果为空并打
`DIV_BY_ZERO` 标记，标记随算子传播到最终输出。合流按批行区间排序拼接；
聚合口径 `none/sum/count/avg` 均为精确整数运算（avg 向零截断，空集为
null），因此任意批大小与任意批执行顺序下最终结果逐字节一致。

## 8. 账本与快照恢复（colval/ledger.py, colval/runner.py）
每完成一批，在**同一 SQLite 事务**写入批标识、结局、内存单元、结果摘要、
结果载荷与进度游标。账本携带整数格式版本（`meta.format_version`）与内容
摘要（全部批记录 + 结果的有序 SHA-256）。重新打开时比较快照版本与当前
版本：不一致则**整体替换**批轨迹与结果（删除全部 batches/results/snapshot，
绝不增量合并），差异以稳定排序（`task#batch`、task_id 排序）写入
`recovery_log`。恢复时依据账本游标只调度未完成批，已完成批从账本载荷
还原参与合流；重放结果与账本记录不一致时以 `E_REPLAY` 定位 task 与最后
提交批。

## 环境说明
- Python 3.10+，第三方依赖仅 numpy；SQLite 走标准库 sqlite3。
- Tkinter 随 Windows python.org 标准发行版自带，`python -m colval.cli gui`
  可用；无 Tk 的平台（如本验证容器）GUI 自动不可用，CLI 不受影响。
