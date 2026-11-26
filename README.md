# MMFM 确定性定点多模态渲染原型

这是从空目录建立的首个可运行版本：本地生成文本、PNG 图像和分块运动条件，经条件编码、定点融合、确定性潜变量初始化、逐去噪步生成、线性解码和分块行 PNG 输出，形成完整渲染管线。所有缓存键、预算调度、快照提交和恢复都以内容指纹为依据；不依赖 Docker、网络服务、真实账号或外部数据集。

本版实际验收结果为 **PASS**，摘要见 [`reports/verification.md`](reports/verification.md)，机器可读明细见 [`reports/verification.json`](reports/verification.json)。

## 目录

```text
src/mmfm/                 渲染库与 CLI
configs/                  JSON/YAML 渲染配置与受控词表
samples/                  固定种子 PNG、运动表、条件清单、编辑流、v1/v2 样本
samples/generate_samples.py
                          样本重生成入口
tests/                    pytest 验收测试
scripts/run_verification.py
                          端到端验证并写 reports/
snapshots/                原子提交的快照根目录（内容可再生，默认忽略）
reports/                  已记录的真实验证结果
```

## 环境

核心渲染要求 Python 3.10+ 和 numpy；PNG 编解码使用标准库 + numpy，不需要 Pillow。

```bash
python3 --version
python3 -c 'import numpy; print(numpy.__version__)'
```

有 sudo 的 Ubuntu/WSL 可安装可选 Tk 依赖：

```bash
sudo apt-get update
sudo apt-get install -y python3-tk
python3 -c 'import tkinter; print(tkinter.TkVersion)'
```

当前评测容器使用独立 Python 3.11.0rc1，numpy 为 1.26.4；其 `tkinter` 缺少系统 `libtk8.6.so`，且当前用户无 sudo。核心渲染与全部验收不依赖 Tk，验证报告如实记录了该项不可用。若能获得系统权限，上述安装命令即可补齐。

无需安装包也可直接从源码运行：

```bash
export PYTHONPATH="$PWD/src"
python3 -m mmfm.cli --help
```

也可执行 `python3 -m pip install -e .` 安装 `mmfm` 命令。

## 快速开始

生成确定性样本：

```bash
PYTHONPATH=src python3 samples/generate_samples.py
```

首次全量渲染前清空快照根目录：

```bash
rm -rf snapshots/*
```

校验输入：

```bash
PYTHONPATH=src python3 -m mmfm.cli validate \
  --config configs/render.json \
  --conditions samples/conditions.json
```

全量渲染：

```bash
PYTHONPATH=src python3 -m mmfm.cli render \
  --config configs/render.json \
  --conditions samples/conditions.json \
  --snapshot-dir snapshots/demo \
  --budget 100 \
  --output-dir reports/demo/full
```

在已提交快照上执行选择性图像替换：

```bash
PYTHONPATH=src python3 -m mmfm.cli render \
  --config configs/render.json \
  --conditions samples/conditions.json \
  --snapshot-dir snapshots/demo \
  --edits samples/edits/selective_image.json \
  --budget 100 \
  --output-dir reports/demo/edited
```

查看提交状态、待办与游标：

```bash
PYTHONPATH=src python3 -m mmfm.cli status --snapshot-dir snapshots/demo
```

预算耗尽后继续：

```bash
PYTHONPATH=src python3 -m mmfm.cli resume \
  --snapshot-dir snapshots/demo \
  --budget 100 \
  --output-dir reports/demo/resumed
```

完整本地验收：

```bash
PYTHONPATH=src python3 scripts/run_verification.py
PYTHONPATH=src python3 -m pytest -q
```

## 输入格式

### 渲染配置

`configs/render.json` 与内容等价的 `configs/render.yaml` 都可加载。关键字段：

- `frame_count`、`denoise_steps`、`chunk_rows`：必须为正整数；`chunk_rows` 不得超过图像高度。
- `condition_dim`、`latent_dim`：条件向量与潜变量维度。
- `image.width/height/mode/tile_size`：固定分辨率和像素模式。模式只接受 `RGB`/`RGBA`，PNG 必须为 8-bit、非交错、支持颜色类型 2/6。
- `fusion_weights_permille`：必须恰好包含 `image`、`motion`、`text`，按类型名升序融合。
- `quant_profiles`、`allowed_quant_profiles`、`allowed_fraction_bits`：量化名字、位宽与枚举约束。
- `fixed_point.milli_scale` 与 `allowed_milli_scales`：交换层千分比定点标尺及允许枚举。
- `noise_scheme`、`noise_schemes`：v2 命名噪声方案与 v1 整数码表。
- `exchange_version`：`v1` 或 `v2`，进入每级缓存键。
- `max_workers`：线程池硬上限；`max_denoise_points`：单批重算去噪点预算，`null` 表示无限。
- `vocab`：受控词表路径，相对配置文件解析。

### 受控词表与条件

词表按 `terms` 顺序定义词项、类型、条件维度和整数权重。提示词只允许 `entities` 与 `attributes` 中的词项；出现未知词或类型不符时返回 `E_UNKNOWN_TERM`。文本编码按词表顺序生成归一化词袋向量，并记录每个命中词项的位置与权重。

`samples/conditions.json` 定义共享的文本、图像、运动资源库，再由帧引用资源 ID。样本中帧 2、3 共享 `img_shared`，因此替换一次图像只会影响引用它的帧；帧 1 保持复用。

图像条件由固定种子绘制几何图元与 8x8 纹理块；编码器按图像块整数统计颜色均值与水平/垂直梯度，再映射到 `condition_dim` 维。运动表要求分辨率、块大小、块行列数与配置完全相容。

### 编辑命令

`samples/edits/*.json` 是命令流对象或数组。已支持：

- `set_denoise_steps`、`set_chunk_rows`
- `set_quant_profile`、`set_exchange_version`
- `set_fusion_weight`（`condition_type` + `value_permille`）
- `set_image` / `set_image_path`、`set_motion` / `set_motion_path`
- `set_prompt`、`assign_frame`、`set_guidance_permille`
- `set_max_workers`、`set_budget`

所有命令先在内存副本上整体应用并重新校验；未知命令返回 `E_UNKNOWN_COMMAND`，非法参数返回 `E_COMMAND_INVALID`，不会留下半应用状态。可用 `mmfm edit` 将验证后的配置/条件写回磁盘。

## v1 / v2 交换

### v1 定宽文本

```text
# MMFM-V1 fields=frame:4,guidance_permille:5,noise_code:2,quant_bits:2
# stage=denoise
0001|01235|01|08
0002|00800|02|12
```

每行固定记录帧号、千分比引导强度、整数噪声码、整数量化位宽，总宽 16 字符。数值超出字段宽度时返回 `E_V1_WIDTH`；字段缺失用空格保留并写入 `missing_fields`。

### v2 JSON

v2 记录浮点引导强度、命名噪声方案、命名量化档位，并保存阶段、帧、步、引用和摘要。Schema 为 `mmfm-stage-v2`。

### 无损往返与旁路

`v2 -> v1` 使用 `ROUND_HALF_UP` 还原千分比，同时把原始浮点值、截断残差、精度标签、阶段/步/摘要及被覆盖引用写入与 v1 文本 SHA-256 绑定的 sidecar。`v1 -> v2` 若 sidecar 有效，优先恢复原始浮点和引用，不因千分比截断丢信息。样例：

```bash
PYTHONPATH=src python3 -m mmfm.cli convert \
  --config configs/render.json \
  --source-version v2 \
  --input samples/exchange/denoise_v2.json \
  --output /tmp/denoise_v1.txt

PYTHONPATH=src python3 -m mmfm.cli convert \
  --config configs/render.json \
  --source-version v1 \
  --input /tmp/denoise_v1.txt \
  --sidecar /tmp/denoise_v1.txt.sidecar.json \
  --output /tmp/denoise_v2_restored.json
```

`validate-exchange` 默认把缺失枚举视为错误；只有显式 `--allow-missing` 才允许部分记录用于诊断。

## 定点渲染与追踪

条件编码先得到千分比整数向量，再量化到所选 `q4/q8/q12` 二进制档位。融合严格按条件类型名升序、再按维度序号升序处理；每次权重乘法和量化都记录 `quantization_step_scale`、整数残差与舍入方向。切换档位会改变输入整数、融合整数和残差总量，不只是改变标签。

初始潜变量由融合摘要、全局种子、帧偏移、配置指纹和交换版本通过 SHA-256 计数器 PRNG 派生，不使用 `random` 或 numpy 全局随机状态。每步固定运算顺序：

1. 按千分比缩放融合条件；
2. 加入该帧/该步噪声表条目；
3. 以整数衰减和半入舍入得到下一步潜变量。

每步保存输出摘要、每个维度的舍入方向/残差、噪声条目摘要和条件引用。帧解码产物保存去噪步链；`trace-diff` 可比较两个快照中的同一帧并给出首个不同步号：

```bash
PYTHONPATH=src python3 -m mmfm.cli trace-diff \
  --snapshot-a snapshots/verification/full \
  --snapshot-b snapshots/verification/incremental \
  --frame 2
```

线性投影使用配置种子生成的固定整数系数。像素计算、半入舍入和 0-255 裁剪全部为整数路径。分块任务只调用请求行区间的投影切片；越界在访问前以 `E_DECODE_OOB` 拒绝。`chunk_rows` 会真实改变行区间、任务数和规范化完成顺序。

## 内容键、失效与账本

每级缓存键包含阶段/算法版本、上游内容摘要、参数摘要、量化档位与交换版本：

```text
condition -> fusion -> latent_init -> denoise(step) -> frame_decode -> chunk
```

账本逐条记录：

- `cache_key`、上游依赖键、参数摘要和产物摘要；
- `reuse_hit` 或 `recompute`；
- 命中来源 `hit_origin=committed`（已提交快照）或 `current_run`（本批刚生成内容）；
- 失效依据，例如 `condition_source_changed`、`upstream_content_changed`、`parameter_changed`；
- 帧的去噪步引用，以及分块行的 `frame_refs` 和 `row_range`。

因此替换共享图像时，只有引用该条件的融合、步、帧和分块失效；帧 1 直接读取已提交产物。调整融合权重则按引用关系使全部帧下游失效。调整分块行数只重算分块，潜变量和帧像素不变。

## 调度、预算与确定性

- 去噪步按稳定键 `(denoise, step, frame)` 组成波次，互不依赖的帧步并发执行。
- 分块任务按 `(decode_chunk, frame, chunk_index)` 稳定排序。
- 线程池容量严格来自 `max_workers`；结果总在回收后按稳定顺序写账和合并。
- 每个实际重算的去噪步消耗 1 点预算。预算在提交任务前按稳定顺序预留，因此完成时机不影响扣减序列。
- 缓存命中不消耗预算；未获预算的步保留为 `E_BUDGET_PENDING` 待办，CLI 退出码为 3。
- 当前运行也记录已提交游标、下一待办步、已解码帧、已完成分块行和累计重算点数。

`MMFM_TASK_DELAY_MS` 只用于验证时让极短任务产生可观测重叠，不进入缓存键、预算或像素。验收中串行观测并发为 1、`max_workers=4` 时为 3，预算序列和待办逐字段一致。

## 快照与强杀恢复

快照根目录结构：

```text
snapshots/demo/
  COMMITTED                       原子替换的提交指针
  .lock                           提交进程锁
  gen-000001/
    manifest.json                 全局版本、配置哈希、输入哈希、父代、state 哈希
    state.json                    产物、账本、游标、待办、预算、workspace
    outputs/frame_0001/frame.png
    outputs/frame_0001/chunk_00.png
  gen-000002/ ...
```

提交步骤：

1. 在 `.tmp-gen-...` 新目录完整写出 state、manifest 和已完成输出；
2. `fsync` 后把临时目录整体改名为新代目录；
3. 最后原子替换 `COMMITTED` 指针。

恢复只读取 `COMMITTED` 指向的代并校验 manifest/state SHA-256、配置哈希和输入哈希。未进入指针的新代、临时目录或未提交阶段不会被采用；`status` 以 `E_SNAPSHOT_INCOMPLETE`（退出码 4）标记崩溃后未完成状态，并报告孤立代 `E_SNAPSHOT_UNCOMMITTED`。

验证用故障注入变量 `MMFM_FAULT_AFTER_COMMIT=N` 会在第 N 次已提交快照后执行真实 `os._exit(137)`。验收记录了强杀返回码 137、停在 `gen-000003`，重启只重算剩余 3 步，最终三帧像素与全量渲染一致。

## 错误码

主要业务码：

| 类别 | 代码 |
| --- | --- |
| 输入/配置 | `E_CONFIG_MISSING`、`E_CONFIG_INVALID`、`E_UNKNOWN_TERM`、`E_DIMENSION` |
| 图像/运动 | `E_PNG_SIGNATURE`、`E_PNG_DIMENSION`、`E_PNG_MODE`、`E_PNG_CORRUPT`、`E_MOTION_DIMENSION`、`E_MOTION_INVALID` |
| 参数枚举 | `E_FRAME_COUNT`、`E_DENOISE_STEPS`、`E_CHUNK_ROWS`、`E_FIXED_SCALE`、`E_QUANT_PROFILE`、`E_ENUM_MISSING`、`E_ENUM_INVALID`、`E_EXCHANGE_VERSION` |
| 交换 | `E_V1_WIDTH`、`E_V1_PARSE`、`E_V2_PARSE`、`E_SIDECAR` |
| 编辑/解码 | `E_UNKNOWN_COMMAND`、`E_COMMAND_INVALID`、`E_CONDITION_REF`、`E_DECODE_OOB` |
| 调度/快照 | `E_BUDGET_PENDING`、`E_SNAPSHOT_VERSION`、`E_SNAPSHOT_INCOMPLETE`、`E_SNAPSHOT_CORRUPT`、`E_SNAPSHOT_UNCOMMITTED` |

CLI 始终输出 JSON 错误对象并以非零码退出。`E_BUDGET_PENDING` 为 3，快照类为 4，其余输入/业务错误通常为 2。

## 已记录验收

`scripts/run_verification.py` 于当前环境产出：

- v2→v1→v2 字段无损，sidecar 重生成与 SHA 校验通过。
- 全量 3 帧、9 个去噪步、每帧 2 个分块完成。
- 替换共享图像后仅帧 2/3 各重算 3 步，共 6 点；帧 1 全部复用。
- `chunk_rows=2 -> 3` 后去噪重算 0 点，行区间变为 `[[0,10],[10,21],[21,32]]`，帧像素不变。
- 2 点预算留下 7 个待办并以 `E_BUDGET_PENDING` 退出；恢复本批重算 7 点后与全量逐像素一致。
- 1/4 线程预算扣减序列、待办和已完成帧摘要一致；并行观测并发 3，未超过 4。
- `os._exit(137)` 后仅从已提交代恢复，最终帧像素与全量一致。
- pytest 共 29 项测试通过。

本版只使用本机确定性样本和本地文件系统，不访问外部目标，不使用容器。
