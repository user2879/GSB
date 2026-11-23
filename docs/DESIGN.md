# 升级仲裁应用技术设计

## 1. 目录与模块

```
escalation_arbiter/
  errors.py         业务错误码与确定性失败清单
  fixedpoint.py     定点整数、千分比、半衰期与半向上舍入
  models.py         证据/工单/规则包/配置/阶段产物数据模型
  injection.py      注入计划（四类真实输入故障）
  loader.py         JSONL 逐行加载与规则/配置/命令校验
  fingerprint.py    规范化 JSON 指纹与阶段缓存键
  references.py     类型/角色/关系条目 -> 工单集合的反向引用
  stages.py         衰减打分、关系扣减加成、仲裁裁决
  cache.py          线程安全的分阶段内容寻址持久化缓存
  engine.py         增量失效、线程池、快照、预算、诊断
  serialization.py  三版本读写/互转/往返统计
  versions/v1.py    定宽文本行
  versions/v2.py    JSON（精确整数 + 浮点视图 + 精度标签）
  versions/v3.py    分区头 + 偏移表 + 定点得分表 + 原因码位图
  diagnostics.py    numpy 偏差直方图与原因码还原统计
  samplegen.py      确定性样本生成
  app.py / __main__.py  命令行编排
tests/              15 个 unittest/pytest 用例
samples/            确定性生成的本地样本
```

## 2. 定点运算约定

- 所有权重、角色可信度、标注可信度、扣减/加成强度、阈值均为**千分比整数**。
- 小数位宽 `decimals` 决定定点网格：`scale_factor(x, d) = round_half_up(x * 10^d / 1000)`。
- 乘法统一走 `mul_permille`，十进制半向上舍入（负数也对称），全程不使用二进制浮点聚合。
- 半衰期系数 `1000 * 0.5^(delta/half_life)` 用 `decimal.Decimal` 50 位精度计算，
  切换半衰期会真实改变系数与最终得分。
- 固定因子顺序：`weight -> role -> confidence -> decay`，证据按序号升序处理，
  每步中间值与舍入方向（exact/up/down）逐条落盘。
- 量化档位 `quantum` 对最终证据得分做半向上对齐，并记录有符号截断残差。
- 互斥扣减后通过 `clamp_low` 保证不低于 `zero_floor`；支持加成不受零下界裁剪。

## 3. 关系与原因码

- 互斥条目按表序、证据对按 `(left_seq, right_seq)` 固定顺序施加扣减；
  支持关系同样顺序施加加成；每条贡献保存前后值与增量。
- 原因码集合：互斥码 `MX<rid>` 在前、支持码 `SP<rid>` 在后，各自按条目编号升序。
- 融合方式 `sum`/`mean` 产出工单裁决分；阈值采用“达到即升级”，
  裁决分恰等于阈值时置 `boundary=true` 且按该档优先级决断。

## 4. 三版本表达力与旁路

| 能力 | v1 定宽文本 | v2 JSON | v3 二进制 |
| --- | --- | --- | --- |
| 定点裁决分 | 定宽整数列，残差入旁路 | 精确整数 + 浮点视图 | 64 位定点整数 |
| 来源角色 | 3 字母别名，未知角色记 `UNK` | 完整角色名 | 长度前缀完整角色 |
| 原因码 | 仅 6 个固定槽位 | 完整有序码 + 命名字典 | 64 位位图，超界入旁路 |
| 精度标签 | 旁路保存 | 内置字段 | 头部保存位宽/档位 |

所有不可表达字段写入对应 `sidecar_v*.json`，逻辑裁决分在三个版本往返后偏差恒为 0，
旁路命中明细进入诊断报告。版本号是 conversion 阶段缓存键的组成部分，切换版本只令
conversion 阶段失效。

## 5. 指纹、缓存与精确失效

- 四级阶段键：`decay -> relation -> arbitration -> conversion`，
  由上游内容指纹、规则参数指纹、原因码字典指纹与交换版本号派生。
- 反向引用索引记录每个证据类型、角色、关系条目引用的工单集合。
- 权重/可信度编辑只失效引用工单；关系增删按两端类型求交；原因码、阈值、半衰期、
  位宽等全局参数失效全部工单；单证据替换只失效其所属工单；版本切换不触发重算。
- 未失效工单直接读取持久化阶段产物；缓存未命中才进入重算集合，
  `reused` 与 `recomputed` 清单按工单 ID 排序输出。

## 6. 并发、快照与预算

- `ThreadPoolExecutor` 并发度取配置且上限 16；任务排序键为 `tid|stage`，
  最终所有清单顺序与单线程按工单 ID 排序一致。
- 快照先写 `tmp/` 再以目录级 `os.replace` 整体替换 `snapshot/`；
  manifest 记录全局版本、配置哈希、规则/字典指纹、完成工单与各阶段键。
- 启动恢复以全局版本与配置哈希双重判定，不一致输出 `E_SNAPSHOT_STALE` 并全量重算。
- 命令处理超过 `budget_ms` 时返回最近摘要并带 `E_INTERACT_BUDGET_EXCEEDED`，
  标记 `parameters_stale=true`。

## 7. 注入故障映射

| 注入 | 真实故障行为 | 业务码 |
| --- | --- | --- |
| `timestamp_reorder` | 交换工单内两条证据时间戳造成倒序 | `E_EVIDENCE_INVALID` |
| `relation_self_reference` | 构造两端相同的互斥关系条目 | `E_RULE_CYCLE` |
| `evidence_type_unknown` | 把证据类型替换为权重表外类型 | `E_RULE_UNKNOWN` |
| `fixed_width_overflow` | 传入越界定点位宽（仅配置副本） | `E_FIXED_WIDTH` |

注入在工单深拷贝上生效，不污染规范状态；单工单失败不影响其余工单，
`inj_id` 与业务码的映射写入诊断报告。
