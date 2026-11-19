# 生成式图像编辑可视化与诊断工作台 — 技术说明

## 1. 目标与范围

工作台在纯本地、零外部下载的条件下，运行一条六阶段确定性图像编辑管线，并在
数据规模连续变化时诊断不同算法路径的取舍：

1. 区域解析 `parse`
2. 掩码生成 `mask`（阈值连通 `threshold` / 边缘追踪 `edge`）
3. 修复填充 `inpaint`（邻域加权均值 `neighbor_mean` / 方向性插值 `directional`）
4. 属性替换 `recolor`（属性值 + 配置指纹派生的确定性重着色 LUT）
5. 风格映射 `style`（量化查表 `lut` / 分通道定点线性 `linear`）
6. 结果合成 `composite`（掩码内取编辑结果、掩码外取源图）

所有随机量来自 `numpy.random.SeedSequence([seed, sample_index])`，同一种子重复
生成的 PNG/JSON 字节完全一致（已用字节级哈希验证）。

## 2. 目录结构

```
config/workbench.yaml            # 本地 YAML 配置（三档规模、预算、阈值、量化档）
config/example_edit_instruction.json
workbench/
  errors.py                      # 业务错误码与异常
  config.py                      # YAML 加载与全部规则校验
  sample_gen.py                  # 确定性样本生成与逐样本校验
  algorithms.py                  # 两条掩码/两条修复/重着色/两条风格/合成
  strategy.py                    # 策略表与规模自适应选择器
  store.py                       # 内容寻址快照存储与 manifest
  pipeline.py                    # 缓存键、阶段链、增量复用、并发、信号边界
  report.py                      # JSON 报告与可视化诊断产物
  cli.py                         # 命令行入口与退出信号处理
tests/                           # stdlib unittest，23 个用例
samples/ outputs/ snapshots/     # 全部产物落盘目录
```

## 3. 确定性与定点规则

- 定点参数：`SCALE_BITS=8`，`Q=256`；`fp_mul`/`fp_div` 使用对称舍入整数运算，
  修复权重、风格权重、风格强度全部以 Q 定点表示，全程不依赖浮点求和顺序。
- 掩码一律为 `uint8` 整数位图（取值 0/1），记录路径名、参数与每区域/并集面积。
- 重着色 LUT（256 项 RGB）由 `blake2b("recolor:{attribute}:{config_fp}")` 派生；
  风格 LUT 由量化档与确定性种子派生；映射表摘要写入阶段记录。
- 内容指纹统一使用 BLAKE2b-128：数组指纹包含 dtype 与原始字节，对象指纹使用
  排序键的规范化 JSON，Windows/Linux 与串/并行下结果一致。
- PNG 导出参数固定（`compress_level=6`、非 optimize），同像素数组导出字节一致，
  已验证并行与串行最终 PNG 字节哈希相同。

## 4. 策略选择器

`strategy.py` 枚举策略表中全部合法组合：
`mask(2) × inpaint(2) × style(2) × quant_levels`，对每个候选计算

- 单样本估计代价（确定性 cost units，`MS_PER_UNIT=0.01` 换算交互延迟毫秒）；
- 质量分（各路径固定质量表 + 量化档质量）；
- 预算超限惩罚 `overrun * 1000`；
- 最终得分 `quality * quality_weight - penalty`。

预算内取得分最高者；全部超预算时回退到最便宜候选并记录原因。指令中的
`mask_hint`/`repair` 作为硬约束收窄候选。决策记录包含规模指标、全部候选评分、
最终选择与理由，重放同一记录复现同一路径组合。

## 5. 缓存键与增量失效

每个阶段的缓存键由上游内容指纹、参数指纹与量化档派生（见
`pipeline.stage_chain_keys`），链式结构：

```
parse_key   <- 图像哈希 + 指令哈希 + 分辨率 + 最大档轮廓 + 算法版本
mask_key    <- parse_key + 掩码路径 + 阈值参数
inpaint_key <- mask_key + 修复路径
recolor_key <- inpaint_key + 持久配置指纹
style_key   <- recolor_key + 风格路径 + 量化档
composite_key <- style_key + mask_key
```

- 修改指令：指令哈希变化，仅该样本 parse 及其下游全部失效。
- 替换图像：图像哈希变化，仅该样本全链失效。
- 调整量化档：仅 style 与 composite 失效（已测试 parse/mask/inpaint/recolor
  复用计数保持不变）。
- 切换规模档：样本文件按最大档规格一次性生成，档位只决定参与处理的样本条数；
  扩张只计算新样本，收窄修剪 manifest 外样本，共有样本逐阶段精确复用，跨档
  最终像素哈希保持一致。

对象不可变、按哈希寻址，从不覆盖；未失效产物直接读取，复用/重算清单逐样本、
逐阶段写入报告。

## 6. 并发模型与逐字段一致

- 线程池 `ThreadPoolExecutor`，最大并发取 `workers`（可被 `workers_override` 覆盖）。
- 每个任务携带 `sample::stage` 稳定排序键；结果与失败清单统一按样本名规范排序，
  manifest 提交也按该顺序进行，与执行模式无关。
- 阶段耗时的规范字段是确定性 cost units（与线程调度无关）；墙上时间仅作为
  诊断字段放在报告的 `stage_timing.wall_time_ms_*`，不参与一致性比对。
- `--serial-check` 在独立临时快照中真实串行重算，逐字段比较决策、阶段摘要、
  最终像素哈希、失败清单与确定性耗时。

## 7. 快照、恢复与信号

- 所有产物先写同目录临时文件，再 `os.replace` 原子替换；manifest 为单次原子写。
- manifest 记录全局版本号、算法版本、配置哈希、已完成样本、失败清单、各样本各
  阶段产物版本（键 + 版本）与历次 run 记录。
- 全局版本号仅在产物实际变化时递增；纯复用 run 不产生版本变化（替换前比对键，
  而非逐文件覆盖）。
- 启动以配置持久指纹 + 算法版本判定续跑，不一致返回 `E_CONFIG_MISMATCH` 并拒绝
  续跑；`active_tier`、`quant_override`、`workers_override` 与本地目录属于运行时/
  环境字段，不参与持久指纹。
- 收到 SIGINT/SIGTERM 时在样本/阶段边界完成当前原子替换与进度记录后退出，
  进程退出码为 0，run 记录标注 `interrupted` 与信号名；下次启动可续跑。

## 8. 业务错误码与失败隔离

| 代码 | 触发条件 |
| --- | --- |
| `E_CONFIG_INVALID` | YAML 结构、并发数、档位单调性、量化档、预算等配置规则不合法 |
| `E_SAMPLE_INVALID` | PNG 容器头/载荷损坏、指令结构、属性值、风格强度非有限数等 |
| `E_REGION_OUT_OF_BOUND` | 区域矩形越界或尺寸非法 |
| `E_CONFIG_MISMATCH` | 快照配置/算法版本与当前配置不一致，拒绝续跑 |

单样本异常被捕获进入失败清单，不中断其余样本；失败样本从完成列表与样本表移除，
修复后下一次 run 自动恢复。

## 9. 报告与可视化

每次 run 在 `outputs/<run_id>/` 生成：

- `report.json`：策略决策、各阶段确定性耗时、复用/失效计数、并发比对摘要、
  失败清单业务码、可复现校验摘要（含报告指纹）。
- `dashboard.png`：决策、复用、失败与阶段代价柱状图。
- `sample_strip.png`：源图/编辑结果对比条带。
- `finals/`：逐样本最终 PNG；`index.html`：静态浏览索引。

## 10. 运行方式

```powershell
# Windows PowerShell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pillow pyyaml
python -m workbench.cli --config config/workbench.yaml --reset --serial-check
```

```bash
# Linux/macOS（本仓库实际验证环境）
python3.11 -m venv .venv
source .venv/bin/activate
pip install numpy pillow pyyaml
python -m workbench.cli --config config/workbench.yaml --reset --serial-check
python -m unittest discover -s tests -v
```

常用参数：`--mode serial|parallel`、`--reset`（忽略快照全量重建）、
`--serial-check`（独立快照串行复核并写入并发比对）。

## 11. 验收结果摘要

- 23 个 unittest 全部通过；`pyflakes` 无告警。
- 多档（small/medium/large）策略随预算切换：宽预算选高质量
  `edge|directional|linear|q64`，紧预算回退 `threshold|neighbor_mean|lut|q64`。
- 编辑序列（档位收窄/扩张 + 指令修改 + 单图替换 + 量化切换）增量结果与空白工作区
  全量重建的最终像素哈希、阶段摘要、决策记录完全一致。
- 并行与串行在独立快照中最终 PNG 字节哈希、像素哈希与规范字段完全一致。
- 失败样本进入清单且其余样本继续；SIGTERM 在下一样本边界退出码 0，续跑补齐。
