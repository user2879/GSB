# 光伏红外组件交换与诊断账本说明

## 1. 交付内容

- `pv_ir_ledger/`：样本生成、输入校验、伪彩解码、栅格统计、边框回归、定点量化、等级判定、v1/v2 交换、内容指纹缓存、预算调度、原子快照和诊断器。
- `tests/test_pv_ir_ledger.py`：12 个 `unittest` 用例，覆盖校验、交换往返、定点量化、多序列增量一致、预算恢复和静默单元隔离。
- `samples/`：固定种子生成的 3 张 320×240 RGB PNG、阵列真值、识别配置、命令流、v1/v2 判定、旁路元数据和 8 组注入清单/文件。
- `reports/diagnostic_report.json` 与 `reports/diagnostic_report.md`：实际端到端验证报告。
- `requirements.txt`：NumPy 与 Pillow。

## 2. 运行方式

Windows PowerShell 或 CMD：

```bat
py -3.11 -m pip install -r requirements.txt
py -3.11 -m pv_ir_ledger generate-samples --root samples
py -3.11 -m pv_ir_ledger verify --workspace . --report reports\diagnostic_report.json
py -3.11 -m unittest discover -s tests -v
```

常用运行：

```bat
py -3.11 -m pv_ir_ledger run --input samples --output runs\current --fresh
py -3.11 -m pv_ir_ledger run --input samples --output runs\current
py -3.11 -m pv_ir_ledger validate-stream --stream samples\command_stream.json
```

第二次执行 `run` 会读取 `runs/current/state/current.json`，按内容键复用未失效阶段。工作区临时目录为 `./.pv_tmp/`，快照先写入该同盘临时目录，再用 `os.replace` 原子替换提交。

## 3. 错误码与隔离

| 错误码 | 触发位置 | 隔离语义 |
| --- | --- | --- |
| `E_IMAGE_FORMAT` | PNG 签名/IHDR、固定 320×240、RGB 模式、Pillow 解码、伪彩色表范围 | 当前图失败，其余 PNG 继续 |
| `E_TRUTH_INVALID` | 真值 JSON、行列数、序号严格递增、图内边框、真值栅格归属 | 当前真值阵列不参与对照，其余阵列继续 |
| `E_CONFIG_INVALID` | 配置和命令流字段、枚举、百分位、阈值、位宽/档位、并发上限 | 返回全部字段问题；非法配置不进入识别 |
| `E_FIELD_TRUNCATED` | v1 行宽不足、字段缺失、数字非法；v2 等级/边框类型错误 | 保留统一记录、缺失标记和原始字段 |
| `E_CELL_DEGRADED` | 非零热覆盖不足或整单元全静默 | 保留单元并给 `DEGRADED`/码 9，不并入邻格 |
| `E_SAMPLE_BUDGET_PENDING` | 单批采样点数预算不足 | 稳定待办队列入快照，下批优先续算 |

## 4. v1 定宽交换格式

每行 72 字符，不写表头；图像名左对齐，数字右对齐，空字段使用 `-`。

| 字段 | 起始列（0 基） | 宽度 | 含义 |
| --- | ---: | ---: | --- |
| `image` | 0 | 32 | PNG 图名 |
| `component_id` | 32 | 6 | 组件序号 |
| `x1` | 38 | 6 | 整数左边 |
| `y1` | 44 | 6 | 整数上边 |
| `x2` | 50 | 6 | 整数右边（右/下边界为开区间） |
| `y2` | 56 | 6 | 整数下边 |
| `intensity_permille` | 62 | 6 | 千分比整数烈度 |
| `grade_code` | 68 | 4 | 整数等级码 |

等级码表 `PV-GRADE-TABLE-V1`：

- `0=NORMAL`
- `1=WATCH`
- `2=MINOR`
- `3=MAJOR`
- `4=CRITICAL`
- `9=DEGRADED`

v2 文档 schema 为 `pv-ir-judgments-v2`，边框是 4 个浮点数，`intensity` 是 0–1 浮点烈度，等级使用命名值，量化方案使用名称，例如 `PV_Q3_L128_UNIFORM_FLOOR_V1`。

## 5. 往返与旁路

- v1→v2：整数坐标转成精确浮点值，千分比烈度除以 1000，整数码映射命名等级；无旁路时量化方案标为 `PERMILLE_Q3_V1`，并加 `EXACT_INTEGER_TO_FLOAT` 精度标签。
- v2→v1：`Decimal` 半入到整数，保存原始浮点文本、坐标残差、烈度残差、精度标签和被丢弃的命名量化方案到 `judgments_v1.sidecar.json`。
- 再次 v1→v2：按 `image#component_id` 找旁路并恢复原始 v2 浮点值、等级和量化方案。
- 诊断器对 72 条样本执行 v2→v1+旁路→v2，规范化 `Decimal` 尾零后逐字段比较逻辑表示，结果为 0 个不一致。

统一表示同时保留 `source_version`、`missing_fields`、`raw_fields`、`precision_labels` 和 `sidecar_key`。解析器来源与原始文本属于取证元数据；参与计算的规范逻辑视图使用相同的十进制值、千分比烈度和等级码。

## 6. 识别链路与固定规则

1. 栅格：边界为 `floor(i * dimension / count)`，区间左/上闭、右/下开；边界坐标归右侧或下侧单元，规则版本为 `FLOOR_EQUAL_SPLIT_LCHO_RCHO_V1`。
2. 百分位：使用 nearest-rank，秩为 `ceil(p * N / 1000)`，取排序后实际热度码。
3. 连通域：4 邻接最大热点连通块；热点掩码为 `heat >= high_percentile_heat`，等于百分位值时包含并标记 `percentile_tie_boundary`。
4. 面积边界：面积千分比等于 `min_hotspot_area_permille` 时标记 `area_tie_boundary`。
5. 覆盖：覆盖率是非零热数据像素占比；低于 `min_coverage_permille` 或整格热度为 0 时输出 `E_CELL_DEGRADED`。
6. 边框：由热点加权质心、连通域外接范围和固定扩展量回归，最终强制裁剪到本栅格单元；`clip_rule=CLIP_TO_EXACT_GRID_CELL;NO_CROSS_CELL_PIXEL_BORROW`。
7. 烈度定点合成，严格按以下整数顺序：

```text
thermal_span = max(0, high_percentile_heat - low_percentile_heat)
sum = thermal_span * 65 + hot_area_permille * 25 + fill_permille * 10
raw_intensity_permille = sum // 100
raw_fixed = round_half_up(raw_intensity_permille * 10^decimals / 1000)
step_fixed = ceil(10^decimals / (quant_levels - 1))
quant_integer = floor(raw_fixed / step_fixed)
intensity_fixed = quant_integer * step_fixed
residual_fixed = raw_fixed - intensity_fixed
```

切换 `fixed_decimals` 或 `quant_levels` 会改变缩放域、步长、量化整数和重建定点烈度；诊断步骤 `QUANT_Q4_L32` 与 `QUANT_Q2_L16` 均断言实际值发生变化。

8. 等级：`intensity_permille >= threshold` 即升入更高等级；恰等于阈值时采用高一级并写 `threshold_boundary=true` 和 `boundary_threshold_permille`。

## 7. 内容指纹与精确失效

缓存键由图像 SHA-256、参数指纹、码表版本和量化档位派生：

- `cell:`：图像指纹 + 行列/分辨率/切分规则 + 行列号和整数区间。
- `stats:`：单元键 + 百分位、覆盖/面积阈值和统计规则。
- `reg:`：统计键 + 边框扩展参数和回归规则。
- `int:`：回归键 + 位宽、量化档位/方案和合成规则。
- `grade:`：烈度键 + 阈值、`PV-GRADE-TABLE-V1` 和等阈值规则。
- `conv:`：等级键 + 交换版本和转换规则。
- `arr:`：排序后的组件等级键集合。
- `img:`：阵列键 + 图像指纹和图级结论规则。

每条 `ledger_events` 记录 `image`、`component_id`、`stage`、`action=REUSED|RECOMPUTED`、`reason`、`cache_key` 和稳定 `event_order`。组件的 `refs` 保存单元区间、统计键和热度统计集合；阵列结论保存排序后的 `component_refs`。

参数影响：

- 交换版本：只失效 `conv:`。
- 阈值：复用栅格、统计、回归和烈度，只重算等级、阵列和图级结论。
- 位宽/档位：复用边框回归，重算烈度、等级和汇总。
- 百分位：复用栅格，统计及其下游按新键重算。
- 行列数：单元区间键变化，真正变化的单元及其下游重算。
- 单图替换：仅该图 SHA-256 变化；其他图所有阶段命中。
- 单图删除：输出集合移除该图；其余图继续命中。

## 8. 预算、并发和快照

组件成本为 `cell_pixel_count * heat_bin_count`。调度键为：

```text
<image_name>|<component_id:08d>|heat_sampling:v1
```

调度器对键排序后顺序扣减预算；下一个任务成本超过剩余预算时停止，后续任务全部进入 `pending_queue`，并输出一个汇总 `E_SAMPLE_BUDGET_PENDING`。恢复时已完成统计键不再扣预算，待办任务仍是同一稳定序列的后缀，因此跨批次的 `state.budget.deductions` 与一次全预算运行完全相同。

并发使用 `ThreadPoolExecutor`，`max_workers=min(config.max_concurrency, task_count)`，硬上限为 8；结果按稳定键重排，不按完成顺序落账。

快照路径：

- 临时目录：`./.pv_tmp/snapshot-v<version>-XXXX/`
- 当前清单：`runs/<case>/state/current.json`
- 历史版本：`runs/<case>/state/snapshots/snapshot-XXXXXXXX.json`

清单包含 `global_version`、`state_schema_version`、`config_hash`、`colormap_version`、`grade_table_version`、`completed_images` 和 `pending_queue`。启动时比较 schema 版本与配置哈希；哈希变化时保留内容寻址缓存，但所有新参数键未命中并重算受影响阶段。残留 `snapshot-v*` 临时事务目录可由 `ignore_stale_transactions` 清理，不影响已提交的 `current.json`。

## 9. 实测验证结果

以下结果由本工作区实际执行生成：

- `python -m pv_ir_ledger verify --workspace . --report reports/diagnostic_report.json`：28/28 项检查通过。
- `python -m unittest discover -s tests -v`：12/12 用例通过；含 3 组固定种子的任意命令序列对拍，本机实测约 27 秒。
- 样本基线：3 张图、72 个组件；真值等级一致 72/72；临界清单非空；无未处理失败。
- 9 个命令步骤（交换版本、5×7 栅格、百分位、阈值、Q4/L32、Q2/L16、替换 A、删除 C、切回 v1）后，增量账本与同最终输入的空白全量在域 JSON、事件键顺序和旁路 JSON 上全部相等。
- 预算测试验证不同耗尽点均为同一稳定任务序列前缀；快照恢复后的域账本和预算扣减序列与全量一致。
- 注入测试验证截断 PNG、伪彩枚举/像素超表、全静默单元、真值越栅格/越图、非法配置和坏 v1 均输出预期错误码且故障隔离。

说明：本次实际命令运行平台记录在诊断报告中为 Linux/WSL，Python 3.11.0rc1；代码只使用 `pathlib`、临时目录和 `os.replace`，未使用 Docker 或平台专有调用，给出的 `py -3.11` 命令适用于 Windows 10/11。
