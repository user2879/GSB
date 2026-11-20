# 骨骼动画重定向管线 — 技术说明

## 1. 模块划分

| 模块 | 职责 |
| --- | --- |
| `animpipe/errors.py` | 七个业务错误码与结构化上下文 |
| `animpipe/quat.py` | 四元数归一化、符号对齐、slerp/nlerp、角距离 |
| `animpipe/hashutil.py` | 确定性 SHA-256 内容哈希（帧哈希、整段哈希、文件哈希） |
| `animpipe/skeleton.py` | 骨架数据模型、固定种子程序化生成、静止姿态差 |
| `animpipe/animation.py` | 自有 JSON 格式、v1→v2 迁移表、时间戳单调性检查 |
| `animpipe/binary.py` | 本地 BTRJ 二进制轨迹格式（字节序/索引宽度/分量顺序） |
| `animpipe/retarget.py` | 逐帧重定向、残差与全局偏差双路对账 |
| `animpipe/resample.py` | 固定帧率重采样，边界不外插 |
| `animpipe/scheduler.py` | 帧块线程池调度，按键归并，块边界无关输出 |
| `animpipe/budget.py` | 线程/缓冲/单帧预算与固定优先级降级 |
| `animpipe/pipeline.py` | 预算规划 + 块调度 + 合并 + 全局符号对齐 |
| `animpipe/incremental.py` | 失效闭包、通道级缓存、增量重算 |
| `animpipe/verify.py` | 增量与串行路径逐帧对账、首个分歧帧 |
| `animpipe/samples.py` | 确定性样本（骨架、动作、BTRJ 样本、v1 旧档） |
| `animpipe/demo.py` | 可重复演示入口，产出 `report.json` |

## 2. 确定性与并行一致性

1. 所有随机数来自固定种子的 `numpy.random.PCG64`，跨进程/平台稳定。
2. 帧块任务是 `(block_key, frame_range)` 的纯函数，工作线程不共享可变状态。
3. 结果按块起始帧键写入预分配槽位，合并顺序与任务完成顺序无关。
4. 四元数符号对齐只在整段数组合并后执行一次，因此符号选择不依赖块边界。
5. 帧哈希只依赖帧内容与参数版本戳；任何 `(workers, block_size)`
   合法组合下整段内容哈希与逐帧哈希逐字节一致（由
   `tests/test_pipeline.py::test_block_partition_byte_invariant` 守护）。

## 3. 重定向数学

局部映射保持“相对自身静止姿态的运动”：

```
Rt_local[j] = Rt_rest_local[j] * (Rs_rest_local[j])^-1 * Rs_local[j]
```

同时维护第二条独立全局路径（按层级累积同一局部公式，等价于全局
静止差 `Tg * Sg^-1`）。两条路径的测地线角距离即全局姿态偏差；
超过阈值（默认 `1e-6 rad`，浮点累积量级 `3e-8 rad`）抛
`E_RETARGET_DIVERGE` 并定位帧与关节。根位移按平均子骨长比例缩放，
朝向语义不变。

## 4. 资源预算降级规则（固定优先级）

1. `max_workers`：钳制并行度；上限为 1 时降级为纯串行路径。
2. `max_buffer_bytes`：块缓冲超限时块大小逐次减半（最小 1 帧）。
3. `max_frame_cost`：单帧计算单元超限时任何块大小都无法容纳，
   直接拒绝并返回 `E_BUDGET_DENIED`，报告可追溯规则、前后并行度
   与估算缓冲字节数。

降级只改变调度形态，帧数学保持纯函数，输出哈希与无约束运行一致。

## 5. 增量失效闭包

缓存键为 `(块起始帧, 通道)`，通道分 `rotation` / `root` / `time`。

| 变更 | 重定向闭包 | 重采样闭包 |
| --- | --- | --- |
| 目标骨长 | 全部块，仅 `root` 通道（旋转缓存复用） | 全部块，仅 `root` |
| 重采样帧率 | 无 | 新网格全部块，含 `time` |
| 关节关键帧 | 受影响源帧所在块，旋转+根 | 输出网格中时间不早于变更时刻的块，外加左侧括号块 |

实际重算集合与清单不一致时抛 `E_CLOSURE_MISMATCH`；闭包外帧哈希
保持不变。

## 6. 交换格式

- JSON：`schema_version`、`quat_order`、`interp`、`joint_names`、
  `joint_index`、逐帧时间戳/四元数/根位移；v1 缺字段由固定迁移表
  补齐，迁移幂等，未知版本抛 `E_SCHEMA_VERSION` 并指明字段。
- BTRJ：魔数 `BTRJ`、版本、字节序标记、索引宽度（1/2/4/8）、
  四元数分量顺序标记、根通道标志、关节数/帧数与逐帧索引通道；
  索引超宽抛 `E_INDEX_OVERFLOW`。float32 通道往返容差 `2e-6`，
  JSON float64 通道容差 `1e-12`。

## 7. 错误码覆盖

`E_SCHEMA_VERSION`、`E_QUAT_INVALID`、`E_INDEX_OVERFLOW`、
`E_TIME_NONMONOTONIC`、`E_RETARGET_DIVERGE`、`E_BUDGET_DENIED`、
`E_CLOSURE_MISMATCH` 均在 `animpipe/demo.py` 的错误链小节与
`tests/` 中以可重复方式触发，上下文定位到帧、关节、字段或预算规则。
