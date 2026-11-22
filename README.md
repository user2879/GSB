# FixedVideo：本地确定性文生视频 / 图生视频多阶段渲染器

这是一个完全本地、无 GPU、无网络服务的确定性视频渲染演示。它把小规模文生视频管线拆成可审计的五个阶段：

1. 受限词表 + 哈希词袋提示词编码；
2. 由提示词指纹、种子和配置指纹派生的固定随机源初始化潜变量；
3. 定点整数去噪、噪声表叠加、引导项与图生视频早期步替换；
4. 线性投影 RGB 解码、0–255 裁剪、按行分块的并发渲染与 PNG 编码；
5. 多帧横向预览序列合成。

命令行交互层在配置的毫秒预算内返回最新已完成预览或 `E_INTERACT_BUDGET_PENDING` 进度；后台继续串行收敛。所有中间产物使用内容指纹缓存，最终结果可用空白全量渲染、单线程渲染和从已提交快照恢复三条路径交叉验证。

## 快速开始

要求 Python 3.10+、NumPy、Pillow。Windows 下将下面的 `python3` 换成 `python`。

```bash
python3 -m pip install numpy pillow
python3 scripts/generate_sample.py --out samples --seed 20261123
```

一次运行完整验收演示：

```bash
python3 scripts/run_demo.py --samples samples --workdir runs/demo-verification --force
```

单独执行交互命令流：

```bash
python3 -m fixedvideo run \
  --config samples/render_config.json \
  --commands samples/commands.json \
  --workdir runs/interactive
```

## 命令行

### `run`

```text
python3 -m fixedvideo run --config CONFIG --commands COMMANDS --workdir WORKDIR
                         [--resume] [--expected-version N]
                         [--stop-after N] [--hard-crash-after N]
```

- `--resume`：从 `WORKDIR/snapshots/CURRENT.json` 指向的已提交代次恢复。配置哈希、命令流哈希、快照 schema、投影指纹或显式版本任一不符，输出 `E_SNAPSHOT_STALE`，退出码 3，不静默续跑。
- `--expected-version N`：恢复时额外要求清单全局版本等于 `N`。
- `--stop-after N`：处理 N 条新命令后完成当前阶段、写快照并退出，退出码 130，用于模拟可处理中断。
- `--hard-crash-after N`：处理 N 条命令后不再提交当前代次，直接 `os._exit(137)`；只能从之前按 `snapshot_interval_commands` 提交的完整快照恢复。

硬终止与恢复示例：

```bash
python3 -m fixedvideo run --config samples/render_config.json \
  --commands samples/commands.json --workdir runs/recovery --hard-crash-after 8
# 退出码 137。快照间隔为 3，因此最后一次完整提交是第 6 条命令。

python3 -m fixedvideo inspect --workdir runs/recovery

python3 -m fixedvideo run --config samples/render_config.json \
  --commands samples/commands.json --workdir runs/recovery --resume
```

### `verify`

```bash
python3 -m fixedvideo verify \
  --config samples/render_config.json \
  --commands samples/commands.json \
  --workdir runs/verification \
  --force
```

验收器会执行并记录：

- 延迟预算内的 pending / 最新预览语义；
- 任意命令序列后的增量渲染与空白全量渲染逐像素、PNG 字节、逐帧指纹、每步潜变量摘要一致；
- 最终必需产物键集合与内容一致，增量复用项与全量重算项逐项一致；
- 1 线程与多线程结果一致；
- 强制中断后从已提交的第 6 条命令游标恢复，最终仍逐像素一致；
- 配置哈希、显式版本、投影指纹三种 stale 条件全部得到 `E_SNAPSHOT_STALE`；
- 失败命令顺序与纯命令重放一致。

报告路径：`WORKDIR/reports/verification.json`。同一报告的副本也写入 `WORKDIR/verification.json`。

### `inspect`

```bash
python3 -m fixedvideo inspect --workdir runs/interactive
```

打印当前指针、清单、配置哈希、全局版本、命令/步/帧/块游标、缓存计数与输出路径。

## 样本输入

`scripts/generate_sample.py` 使用固定种子生成本地样本，不访问网络：

- `samples/render_config.json`：词表与嵌入维度、种子范围、步数、引导强度、分辨率、分块行数、定点位宽、量化档位、延迟预算、线程数、快照间隔、噪声表版本、重算瓦片预算、图生视频早期替换步数。
- `samples/vocabulary.txt`：有序受限词表。
- `samples/noise_tables/nt-v1.npz`、`nt-v2.npz`：带 `noise`、`version`、`shape`、`dtype` 描述符的 int64 定点噪声表。
- `samples/initial_latent.npz`、`initial_latent_alt.npz`：带 `latent`、`shape`、`dtype` 描述符的 float32 初始图编码潜变量。
- `samples/commands.json`：20 条命令，覆盖提示词、种子、引导强度、步数、帧数增减、初始图替换与移除、补丁增删、分块行数、量化档位，以及两类预期失败。

量化档位映射：`q1=1`、`q2=32`、`q4=256`、`q8=1024` 个定点 LSB。样本位宽为 12 bit，因此 `q4` 的量化间隔为 `256/4096=1/16`。

## 输出目录

每个已提交代次位于 `WORKDIR/snapshots/gen-NNNNNN/`：

```text
manifest.json                 全局版本、配置哈希、命令流哈希、阶段、游标、确定性指纹
state.json                    可恢复的有效渲染状态
cache/                        当前提交状态所需的精确阶段产物与缓存索引
determinism/                  投影矩阵、像素偏置、条件矩阵与指纹
outputs/frames/               最终逐帧 PNG 与解码元数据
outputs/blocks/               每个分块行带的 PNG
outputs/preview_strip.png     横向多帧预览序列
outputs/preview.json          帧顺序、内容指纹、解码参数、块摘要
reports/commands.json         逐命令延迟、命中率、复用/重算键、失败
reports/failures.json         按命令顺序排列的失败清单
reports/latest_preview.json   最近一次完成预览摘要
reports/reuse_recompute.json  逐命令复用与重算清单
```

顶层 `WORKDIR/reports/diagnostics.json` 是每次运行的总诊断报告，包含逐命令响应延迟、收敛延迟、命中率、待重算项、失败清单和最终指纹。

## 错误码

| 错误码 | 含义 |
| --- | --- |
| `E_CONFIG_INVALID` | 配置、命令字段类型、数组描述、枚举或形状不合法 |
| `E_SEED_OUT_OF_RANGE` | 配置或命令种子超出 `seed_range` |
| `E_NOISE_TABLE_MISMATCH` | 噪声表版本、文件版本、形状或 dtype 不符 |
| `E_INTERACT_BUDGET_PENDING` | 预算内未完成最新参数渲染，返回带版本标记的进度 |
| `E_PATCH_OUT_OF_RANGE` | 局部补丁矩形越过潜变量空间边界 |
| `E_SNAPSHOT_STALE` | 恢复版本、配置哈希、命令流、投影或 schema 不符 |

## 测试

```bash
python3 -m pytest -q
```

当前实际结果：`16 passed`。测试覆盖校验错误码、无全局随机依赖、图生视频分支种子敏感性、提示词/初始化复用、补丁精确失效、分块与量化有效性、交互 pending 语义、快照 stale 拒绝，以及完整端到端验收。

## 实测验收摘要

在 Python 3.11.0rc1、NumPy 1.26.4、Pillow 11.0.0 环境执行 `verify`：

- 全部 10 项验收检查为 `true`；
- 交互预算 8 ms，最大实测响应 6.325 ms，20 条命令中 18 条返回 pending；所有 pending 都标注配置版本，第二条起携带最近完成预览；
- 增量最终帧与空白全量渲染：8/8 帧逐像素一致、PNG 字节一致、帧指纹一致、每步摘要一致；
- 最终必需产物 178 个键；增量复用 137 项、重算 41 项，键集合与内容同全量重算完全对应；
- 多线程与单线程结果一致；
- 强制中断后的恢复从命令游标 6 开始，恢复后输出仍逐像素一致；
- 配置哈希、显式版本和投影指纹不匹配均返回 `E_SNAPSHOT_STALE`；
- 失败顺序为 `c19: E_PATCH_OUT_OF_RANGE`、`c20: E_SEED_OUT_OF_RANGE`，与纯重放一致。

更完整的算法、定点公式、缓存键与快照协议见 `docs/TECHNICAL.md`。
