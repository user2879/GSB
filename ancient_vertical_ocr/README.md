# 竖排古籍版面识别（确定性整数内核演示）

面向本地、确定性、可重复演示的竖排古籍版面识别首版。全部数值计算在定点整数域完成，
不访问外网，不依赖容器；样本由固定整数种子与内置点阵字模合成。

## 1. 目录结构

```
ancient_ocr/          引擎包
  errors.py           业务码与带样本标识/位置的校验异常
  fixedpoint.py       定点量化枚举、q 点换算、半向上舍入（记录方向）、整数三角表
  hashing.py          blake2b 规范指纹与缓存键
  rng.py              固定种子 LCG
  fonts.py            18 个内置 12x12 点阵字模
  sample_gen.py       320x480 竖排版面合成、整数旋转/抖动/遮挡、真值与栏序/关键点版本
  config.py           配置加载与严格校验
  loader.py           PNG 头/分辨率/模式、真值、栏序、关键点版本、命令流校验与隔离
  windows.py          滑窗粗定位、定点特征、指纹投影、扫描行预算与待扫续扫
  components.py       连通域、碎片合并、噪声过滤、关键点回归、字模比对账本
  ordering.py         栏内串联、声明冲突、跨栏合并、并列裁决、待裁决业务码
  cache.py            分阶段内容寻址缓存、上游引用、精确失效/复用
  scheduler.py        三类预算联合调度、线程池并发上限、待办优先、预览摘要
  snapshot.py         临时目录 + os.replace 原子提交、全局版本/配置哈希、恢复判定
  pipeline.py         阶段编排与命令流增量/全量统一入口
  cli.py / gui.py     命令行 / Tkinter 诊断台（无 Tk 时优雅降级）
samples/ config/ snapshots/ reports/ commands/ logs/
tests/                21 个 unittest 用例
run_demo.sh           一键端到端演示
```

## 2. 定点整数约定

- 量化档位枚举：`coarse=2 / standard=4 / fine=8 / ultra=16` 位小数，`quant_level` 与
  `frac_bits` 必须一致，否则 `E_CONFIG_INVALID`。
- 所有特征（墨迹密度、竖直投影谷、纹理梯度、候选得分、重心、可信度）均为 q 点整数；
  乘除使用 `FixedPoint.mul/div`，避免浮点。
- 关键点回归采用半向上舍入，逐点记录 `round_x/round_y ∈ {exact, up, down}` 与
  `confidence_q`。
- 旋转使用内置整数度数表（-5/-3/0/3/5 度）的定点 cos/sin，逆映射采样。

## 3. 三类预算与联合调度

| 预算 | 计量口径 | 阻塞阶段 | 耗尽行为 |
| --- | --- | --- | --- |
| 扫描行 | 每窗 `window_size` 行 | 粗定位 | 窗口入待扫队列，保留指纹，恢复后续扫不重扫 |
| 比对点 | `字块墨点数 × 模板变体数` | 细定位/字模比对 | 组件入待办，账本逐条扣减 |
| 栏间裁决点 | 每次并列/候选裁决固定消耗 | 跨栏合并 | 未合并栏转待裁决，输出 `E_ADJUDICATION_BUDGET` |

某类预算不足只阻塞对应阶段，其余阶段继续推进并刷新预览摘要；待办单元携带阶段名与
已消耗预算，下一轮优先处理，不从头重算。线程池 `max_workers` 取自配置且硬上限 8，
结果按 `文件名|阶段|行列/序号` 稳定排序键回排，任意并发度产物一致。

## 4. 缓存键与精确失效

每阶段缓存键由图像内容指纹、参数指纹、量化档位、关键点版本指纹派生；条目记录稳定的
上游引用集合。

- 关键点版本整体替换：识别版本指纹变化，仅失效引用该版本的 `fine/intra/merge`，
  `coarse` 与其他样本保持复用。
- 单图替换：仅该样本 `load/coarse` 及下游失效。
- 滑窗/量化参数变更：仅影响对应参数指纹的阶段。
- 增删样本：按 `(sample, stage)` 条目增删，删除时同步清理待办与失败清单。

## 5. 快照与强制终止恢复

阶段（load/coarse/fine/intra/merge）产物、待办队列、三类预算剩余先写 `snapshots/tmp-*`，
再以 `os.replace` 整体替换 `snapshots/current`，附 `COMMITTED` 标记、全局版本号与
配置哈希。恢复只认真实已提交快照；配置哈希不一致时返回
`E_SNAPSHOT_VERSION_MISMATCH` 并回退全量重建，不做逐文件覆盖。

## 6. 业务码

`E_SAMPLE_INVALID`、`E_CONFIG_INVALID`、`E_KEYPOINT_VERSION_INVALID`、
`E_SCAN_BUDGET_EXHAUSTED`、`E_COMPARE_BUDGET_EXHAUSTED`、`E_ADJUDICATION_BUDGET`、
`E_SNAPSHOT_VERSION_MISMATCH`、`E_RECOVERY_FALLBACK`。

坏样本 `bad_oob`（越界关键点）、`bad_cycle`（环状邻接）、`bad_header`（分辨率/PNG 头）
被隔离为 `E_SAMPLE_INVALID` 并带位置，不影响 `page_a/b/c` 的结论。

## 7. 使用方式

```bash
python3 -m pip install -r requirements.txt
python3 -m ancient_ocr.cli init --root .        # 生成配置与确定性样本（含坏样本）
python3 -m ancient_ocr.cli run --root .         # 全量运行并写 reports/
python3 -m ancient_ocr.cli commands --root . commands/demo_commands.json
python3 -m ancient_ocr.cli crash-demo --root .  # 首次 SIGKILL，再次运行恢复
python3 -m ancient_ocr.cli gui --root .         # 需 python3-tk 与显示环境
python3 -m unittest discover -s tests           # 无显示环境可跑全部引擎测试
```

小预算阻塞可通过 `--overrides` 注入，例如：

```bash
python3 -m ancient_ocr.cli run --root . \
  --overrides '{"scan_row_budget":64,"compare_point_budget":100,"adjudication_budget":1}'
```

## 8. 验证结果

- 每个正常样本稳定得到 40 个字块组件（4 栏 × 10 块），碎片合并逐条记录依据。
- 栏内路径纵坐标单调，跨栏最终顺序 `[0,1,2,3]`（右到左），扁平阅读顺序 40 项。
- 标注关键点 v1 与真值曼哈顿误差为 0；回归逐点带舍入方向与可信度。
- 三类预算账本均有逐条扣减；小预算注入时分别产生待扫/待比对/待裁决与对应业务码。
- 21 个 unittest 覆盖：定点域、配置/样本校验隔离、预算阻塞、缓存精确失效、
  增量与全量逐字段一致、并发度无关性、真实 SIGKILL 后快照恢复。
