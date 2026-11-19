# MocapClean Workbench — 技术说明

## 1. 确定性边界

所有输入均由 `mocap_workbench.generator` 在本地生成，不读取外部动捕资产。固定参数为：

- 帧率 `60 fps`，骨架固定 `19` 个关节，四元数顺序 `w,x,y,z`。
- 帧数仅接受 `1200 / 6000 / 12000`。
- 固定种子默认 `20261120`，事件排程使用自实现的 `SplitMix64`，不依赖 Python `hash()`。
- 数值写入前按 `9` 位小数取整，`-0.0` 规范为 `0.0`。
- JSON 使用 UTF-8、排序键、紧凑分隔符和末尾换行；内容 SHA-256 写在文件内。

生成器先产生平滑的解析型基座姿态，再把备份作为 truth 写出；随后在副本注入：

| 类型 | 注入规则 | 真值清单粒度 |
|---|---|---|
| `joint_jump` | 单帧乘以 48°–72° 额外旋转 | 该帧、该关节 |
| `jitter` | 8–13 帧 Nyquist 正负交替小幅叠加，2.1°–2.9° | 每个被污染帧、关节 |
| `missing_marker` | 3–8 帧写入 `[0,0,0,0]` 占位 | 每个无效帧、关节 |
| `root_slip` | 单帧根位移加 0.24–0.46 m 级瞬时偏移 | 该帧、`Root` |

事件之间保留确定性避让间隔，truth 只进入评估层，修复算术不读取 truth。

## 2. 检测证据与分类

`compute_evidence()` 只依赖原始记录，与阈值无关。每个帧/关节保存：

- `step_prev_deg` / `step_next_deg`：相邻单位四元数夹角。
- `midpoint_residual_deg`：当前四元数与左右邻帧 SLERP 中点的夹角。
- `angular_second_diff_deg`：`q(t-1)^-1 q(t)` 与 `q(t)^-1 q(t+1)` 的相对旋转向量差。
- `invalid_placeholder`：范数是否为 0。
- 根轨迹中点残差与相邻根步长。

分类时先直接标记无效占位为 `missing_marker`；其余连续候选段做峰值归因：

- 段峰值超过 `joint_jump_deg`：只标记峰值帧为 `joint_jump`，消除两侧过渡肩部误报。
- 否则保留峰值的至少 65% 且超过 `joint_jitter_deg` 的连续内部帧为 `jitter`，消除有限抖震块两端的半幅肩部。
- 根残差段归因到峰值帧为 `root_slip`。

该设计使默认阈值下的检测清单与生成器真值清单按 `(frame, joint, type)` 精确一致。检测 JSON 同时保存每个帧、关节和 `Root` 的全部证据量值及真值比对结果。

## 3. 稀疏关键姿态与修复规则

每个关节维护独立锚点流，根轨迹维护独立锚点流。锚点由两部分组成：

1. 检测出的连续缺陷区间两端干净帧（自动锚点，可在时间轴中抑制/移动）。
2. 用户手工添加的全局关键帧；关键帧帧号必须有效、未被检测为缺陷且不得重复。

若某一侧没有锚点，使用唯一可用端点姿态常量延拓。缺陷坐标取左右最近锚点 `a < t < b`，比例 `alpha=(t-a)/(b-a)`：

- 旋转：最短路径 SLERP；若 `dot(q_a,q_b)<0` 先将右端乘 `-1`。
- 根位移：逐分量线性插值。
- 结果归一化并按 9 位小数量化；不对每个输出帧独立做半球规范化，避免人为符号跳变。
- 所有干净帧直接复制原始分量，不会因关键帧存在而被改写。

全量修复与增量修复使用同一个 `repaired_joint_value()` / `repaired_root_value()` 算术，因此同一输入可逐字节复现。

## 4. 误差与平滑性口径

- 四元数误差：`2*acos(min(1, abs(dot(q_repair, q_truth))))`，单位度。
- 根误差：三维欧氏距离，单位米。
- 角加速度：设 `r(t)=log(q(t)^-1 q(t+1))`，在帧 `t` 取 `||r(t)-r(t-1)|| * 60²`，输出 `deg/s²`。
- 每个修复坐标都有明细；每个连续修复区间输出最大值、RMS、原始/修复/真值角加速度峰值和改善比例。
- 可信容差固定为关节角误差 `2.0°`、根误差 `0.02 m`。任一明细超差时该明细 `trusted=false`，总体状态为 `WITHHELD_UNTRUSTED`；明细不会被汇总掩盖。

## 5. 增量失效闭包

依赖图为：

```text
thresholds -> detection -> defect intervals -----------------+
keyframes / auto suppression -> keyframe intervals -----------+-> repair -> evaluation
```

阈值变化先复用证据缓存重新分类；关键帧变化不重算证据。对每个关节：

1. 取旧/新缺陷帧并集。
2. 比较每帧是否废帧、缺陷类型，以及旧/新最近锚点对。
3. 只有上述状态或锚点对发生变化的坐标进入修复闭包。
4. 根轨迹使用同样规则。
5. 评估只重算与脏帧相交（含 ±1 加速度邻域）的区间。

`repair_dirty()` 从旧快照复制整个不可变姿态数组，只覆盖闭包坐标。随后将实际重算坐标与闭包集合精确比较；不相等抛出 `E_CLOSURE_MISMATCH`。测试同时验证闭包外逐帧姿态哈希和区间指标哈希不变。

## 6. 增量与全量一致性

每次候选预览都会先做增量，再以当前阈值/关键帧/自动边界做一次全量修复，逐帧逐关节比较：

- 四元数分量绝对容差 `1e-9`，归一化角差容差 `1e-5°`。
- 根分量绝对容差 `1e-9 m`。
- 不使用 RMS 代替逐项判定。
- 首个分歧按帧号升序、关节名文件顺序扫描，并报告阶段、关键帧区间端点、参数版本戳。

候选快照若不一致，界面禁止提交并显示 `E_CLOSURE_MISMATCH`。

## 7. 并发预览与快照替换

`SnapshotManager` 持有一个已提交快照单指针和一个后台 `ThreadPoolExecutor(max_workers=1)` 候选 future。

- 编辑生成候选时捕获当前不可变快照、内容哈希与版本戳 label。
- 候选完成后同时核对基线内容哈希与版本戳；任一变化即标记 `E_SNAPSHOT_STALE` 并丢弃，内容相同但版本戳不同也不得提交。
- 预览只存在内存；确认提交才写账本并替换单指针。
- 提交前再次同时校验候选记录的基线内容哈希与版本戳，避免版本变化后误用旧候选。
- GUI 每次 `paint`/表格刷新只读取一个局部快照对象，不逐帧混读新旧对象。
- 每帧修复耗时、总耗时、预算和累计超预算次数写入 `runtime/state/<namespace>/performance.json` 以及 `runtime/logs/interaction/`。

声明预算：1200 帧 `900 ms`、6000 帧 `1800 ms`、12000 帧 `3200 ms`。超时仍可查看候选，但结果带 `E_BUDGET_EXCEEDED` 标记且计入超预算次数。

## 8. 原子账本与恢复

提交顺序：

1. 检测、修复、评估和完整 ledger 分别序列化为内容寻址 blob，路径 `runtime/state/<namespace>/blobs/<sha256>.json`。
2. 逐个 blob 使用同目录临时文件、`fsync`、`os.replace` 原子落盘。
3. 最后原子写入 `current.json` 指针；该写入是唯一提交点。

若在指针替换前终止，新 blob 不被任何指针引用，重启时作为孤儿回收。`ledger` 包含版本戳、阈值、关键帧、自动边界抑制、检测清单、修复逐帧哈希、评估明细、闭包与一致性报告。

会话语义：

- 启动写 `session.json`，表示当前会话活跃。
- 正常退出写 `closed.json`（`clean_exit=true`、当前 commit id）并删除 session。
- 若下次启动看到上一 session 但 closed 标记缺失或 commit 不匹配，记录强制终止，不伪造正常退出。
- 恢复只读 `current.json` 指向的 ledger，并按记录输入和提交参数重算、比对内容哈希与修复哈希；不一致抛 `E_SNAPSHOT_STALE`。

## 9. 业务错误码

| 错误码 | 触发位置 |
|---|---|
| `E_FRAME_MISSING` | 文件/帧缺失、帧数不匹配、非法档位 |
| `E_THRESHOLD_RANGE` | 阈值越界、未知阈值键、值非数值 |
| `E_KEYFRAME_INVALID` | 越界、重复、缺陷帧锚点、不可移动目标 |
| `E_QUAT_INVALID` | 非有限值、非单位且非占位四元数、内容哈希不匹配 |
| `E_CLOSURE_MISMATCH` | 实际重算集合 ≠ 闭包，或增量/全量逐项不一致 |
| `E_SNAPSHOT_STALE` | 后台结果基线过期、账本 blob 缺失/损坏、恢复哈希不符 |
| `E_BUDGET_EXCEEDED` | 候选预览超过声明交互预算 |

## 10. 关键源码

- `mocap_workbench/generator.py`：确定性生成、注入、规范 JSON 读写。
- `mocap_workbench/detection.py`：证据计算、分类、真值比对。
- `mocap_workbench/repair.py`：锚点、SLERP/根插值、脏坐标修复。
- `mocap_workbench/evaluation.py`：逐帧误差、加速度、区间汇总、可信标记。
- `mocap_workbench/pipeline.py`：全量/增量闭包、一致性。
- `mocap_workbench/preview.py`：交互预算、后台候选、快照单指针。
- `mocap_workbench/persistence.py`：原子账本、恢复、孤儿回收。
- `mocap_workbench/gui.py`：PySide6 时间轴与工作台。
- `mocap_workbench/verification.py`：三档验收脚本。
