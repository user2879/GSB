# 演奏表达时间轴诊断工作台 — 技术说明

## 1. 总体架构

```
mwb/
  errors.py       业务码 E_xxx 与统一异常
  atomicio.py     规范化 JSON、SHA-256 摘要、同目录临时文件 + os.replace 原子写
  models.py       乐谱 / 表达参数 / 编辑的数据模型与算子白名单
  parse.py        乐谱解析校验（音高、拍位、时值、声部、同声部重叠）
  parse_params.py 速度/力度曲线、伸缩、连断、踏板校验
  parse_edit.py   编辑脚本算子白名单与字段校验
  expand.py       拍位 -> 整数 tick 确定性展开、跨声部重叠区间
  curves.py       速度分段线性插值、力度映射、踏板轨道（NumPy）
  chunkstate.py   单块表达状态计算 + 五类依赖键登记
  editengine.py   编辑逐步重演、目标语义校验、依赖键与 tick 区间失效
  persistence.py  分块持久化、版本目录、摘要校验、孤儿块隔离
  runner.py       有界线程池、内存预算门控、增量重算、信号中断、坏块自愈
  report.py       全量重演基准、容差比对、逐块摘要校验、报告/时间轴联合原子提交
  samples.py      内置确定性样本生成器（无随机源）
  cli.py / __main__.py  命令行入口与退出码
  gui.py          tkinter 只读桌面视图（不重算、不改写已校验块）
```

## 2. 关键设计决策

### 2.1 整数 tick 与唯一取整规则
- `tick = round(beat * ticks_per_beat)`，样本拍位均为整数，结果精确且可复现。
- 音符区间为半开 `[start_tick, end_tick)`；同声部区间相交直接 `E_OVERLAP_SAME_VOICE`。
- 跨声部重叠只记录，按声部对输出合并后的 tick 区间到诊断报告。

### 2.2 表达状态演进
- 速度：控制点拍位换算为 tick 坐标后 `numpy.interp` 分段线性，末端恒定。
- 时值伸缩：`actual_ticks = round(nominal * stretch * bpm_base / bpm_at_start)`。
- 连断：发音比例 gate 决定 `active_end`，剩余为静音段尾；缺省 legato/normal/staccato。
- 力度：映射曲线分段线性后四舍五入并钳制 0..127。
- 踏板：段落换算 tick 后生成 0/1 轨道，逐 tick 挂到音符状态。
- 每个 tick 可查询：活跃/静音、实际力度、当前速度、踏板是否踩下。

### 2.3 依赖键与增量失效
每块登记五类依赖键：`tempo:i`、`velocity:x`、`stretch:name`、
`articulation:name`、`pedal:id`。编辑后：
- 键命中的块标记失效重算，其余块直接复用；
- 新增类编辑（如 `pedal_add`）旧块无对应键，额外用显式 tick 区间判定失效；
- 速度控制点索引可能重排，速度类编辑保守失效全部速度键。

### 2.4 版本识别与恢复
- 版本标识 = 乐谱+表达参数+编辑脚本规范化 JSON 的 SHA-256 前 16 位。
- 每版本独立目录 `state/<version>/`，版本不一致另起目录，绝不覆盖旧版本。
- 块文件含内容摘要；启动时逐块校验，坏块/缺块剔除后回退补算（E_DIGEST 自愈）。
- 强制终止残留的无主块（有文件无 manifest 条目）移入 `orphans/`。

### 2.5 原子性与退出码
- 所有 JSON 先写同目录 `.tmp-*` 临时文件再 `os.replace`。
- 报告与时间轴在同一临时目录备齐后依次替换，失败保留旧版本。
- 退出码：`0` 正常完成，`2` 可处理终止（块边界落盘中断标记），`1` 业务错误。
- 业务错误输出带 `E_xxx` 的 JSON 到 stderr，且不产生任何持久化输出。

### 2.6 并发与内存预算
- `ThreadPoolExecutor` 并发度取编辑脚本声明值；预算信号量限制同时在算块数。
- 预算占满时按块索引序让后续块等待，不降低 tick 分辨率或插值精度。
- 任一块异常取消其余待处理块并抛首个错误；块计算是纯函数，并发度变化逐 tick 等价。

## 3. 业务码清单
| 码 | 含义 |
| --- | --- |
| E_PARSE / E_SCHEMA | JSON 语法 / 结构字段错误 |
| E_PITCH / E_BEAT | 音高越界 / 拍位或时值非法 |
| E_VOICE / E_OVERLAP_SAME_VOICE | 声部标识 / 同声部 tick 重叠 |
| E_TEMPO / E_VELOCITY_CURVE | 速度曲线 / 力度曲线约束非法 |
| E_EDIT_OP / E_EDIT_TARGET | 算子不在白名单 / 引用目标不存在 |
| E_DIGEST / E_VERSION | 摘要或 tick 数不符 / 版本不一致 |
| E_BUDGET / E_IO | 内存预算非法 / 文件 IO 错误 |

## 4. 桌面视图
`python -m mwb.cli gui` 读取 `output/` 下已校验的 timeline 与报告，
展示音符条带（发音段/静音段着色）、速度曲线、踏板轨道、逐步失效列表、
跨声部重叠与摘要结论；点击块高亮其依赖键。视图全程只读，不触发重算。
无 tkinter 环境时返回 E_IO JSON 并以退出码 1 优雅降级，核心流程不受影响。
