# 光谱成像数据处理与可视化诊断工作台

这是一个完全离线、确定性的科学成像诊断项目。输入由固定种子生成为
`64 × 256 × 256` 的 `uint16` 光谱立方体，每个 `256 × 256` 帧对应一个光谱带。
项目完成暗电流/平场校正、端元 NNLS 解混、浓度误差传播、亚像素漂移补偿、
参数化增量缓存、不可变诊断快照与 PyQt5 只读视图。

## 快速开始

目标环境为 Windows + Python 3.10 及以上，只依赖 NumPy 1.24 和 PyQt5 5.15。

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --no-index --find-links .\wheels -r requirements.txt
```

将离线 wheel 放入 `wheels\`。联网环境也可直接
`python -m pip install -r requirements.txt`。仓库不含真实仪器数据。

生成默认全量立方体并运行完整诊断：

```powershell
python -m spectral_workbench generate --out artifacts\demo\cube
python -m spectral_workbench run `
  --cube artifacts\demo\cube `
  --out artifacts\demo\run `
  --snapshot artifacts\demo\snapshot `
  --version 1 `
  --verify-closure
```

查看已发布快照（需要 Windows 图形会话；无桌面环境可设置
`QT_QPA_PLATFORM=offscreen`）：

```powershell
python -m spectral_workbench ui --snapshot artifacts\demo\snapshot
```

运行测试：

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m unittest discover -v
```

也可使用 `scripts\run_tests.ps1` 和 `scripts\run_demo.ps1`。

## 已实现能力

- 确定性生成器：PCG64 固定种子、原始帧/暗场/平场、逐带真值、区域掩膜、
  中心、真实浓度、通道索引、端元谱和逐帧峰值/均值/饱和计数。
- 校正：`(raw-dark)/((flat-dark)/mean(flat-dark))`，输出 `float32`；
  饱和、零平场和不可用像素显式置零并写入掩膜和计数，绝不静默处理。
- 解混：枚举活跃集实现精确小规模 NNLS；所有 64 个光谱带共同构成每个像素的
  光谱，因此 NNLS 是立方体级求解，浓度指标按光谱带逐带汇总。
- 误差传播：每像素每荧光团给出校正量化、残差、条件数放大和浮点舍入四项，
  并合成可复算上下界；报告均值、95 分位、超容差比例和区间覆盖率。
- 漂移：以真值参考中心限定窗口，用信号积分图加权质心估计亚像素平移；
  质量门控后与真值漂移比较，超过容差返回 `E_DRIFT_BIAS`。
- 增量：校正、漂移、NNLS 与逐带指标均使用内容寻址缓存，运行记录逐帧保存
  `computed/reused` 来源和版本哈希；可选全量重算闭合校验定位首个分歧像素。
- 快照/UI：快照数组只读，整体替换必须同时匹配递增版本戳与 SHA-256 内容哈希；
  UI 只渲染通过门控的快照，不引用管线或服务，因此不能触发重算。
- 故障注入：暗电流不一致、端元退化、中心偏移、饱和过载和已知浓度偏差均
  通过可序列化输入参数生效，并覆盖全部业务错误码。

## 关键命令

```powershell
# 生成带已知浓度偏差的本地样本
python -m spectral_workbench generate `
  --out artifacts\demo\injection_case\cube `
  --seed 7 --frames 64 --height 64 --width 64 `
  --injections artifacts\demo\injection_config.json

# 诊断并在报告内定位首个越界像素
python -m spectral_workbench run `
  --cube artifacts\demo\injection_case\cube `
  --out artifacts\demo\injection_case\run `
  --tolerance 0.08 --verify-closure

# 复算指定带/像素的误差分量
python -m spectral_workbench inspect-pixel `
  --cube artifacts\demo\injection_case\cube --run artifacts\demo\injection_case\run `
  --band 0 --y 30 --x 30
```

`run` 支持 `--config config.json`、`--calibration-epsilon`、`--unmix-order`、
`--cond-warn`、`--endmember-file`、`--no-compensate`、`--max-shift` 和
`--tolerance`。参数变化后，相同 `--out` 会复用未失效阶段。

## 目录

- `spectral_workbench/generator.py`：确定性样本与真值。
- `spectral_workbench/calibration.py`：校正和量化误差上界。
- `spectral_workbench/unmix.py`：端元校验、NNLS、条件数与像素误差分解。
- `spectral_workbench/drift.py`：亚像素估计和双线性补偿。
- `spectral_workbench/metrics.py`：浓度误差、不确定度与越界定位。
- `spectral_workbench/cache.py` / `pipeline.py`：增量缓存和完整诊断管线。
- `spectral_workbench/snapshot.py` / `service.py` / `ui.py`：不可变快照与只读视图。
- `docs/technical.md`：公式、缓存边界、错误码和复算说明。
- `artifacts/demo/run/diagnostic_report.json`：默认全量样本的诊断报告。
- `artifacts/demo/VALIDATION.md`：本次实际验证结果与命令。

## 错误码

| 错误码 | 含义 |
|---|---|
| `E_CUBE_FORMAT` | 立方体声明/尺寸/哈希不合法，或饱和比例违反输入策略 |
| `E_DARK_GAIN` | 暗场越界、零平场比例过高、有效平场均值非正 |
| `E_UNMIX_COND` | 端元非有限、零范数、秩亏或条件数超过致命阈值 |
| `E_DRIFT_BIAS` | 估计漂移超过配置容差 |
| `E_CLOSURE_MISMATCH` | 增量结果与全量重算不一致，详情包含首分歧带/像素 |
| `E_SNAPSHOT_STALE` | 快照版本/哈希不是当前版本，或快照文件损坏 |
| `E_INJECT_CONFIG` | 注入/参数配置不可序列化或不满足范围约束 |

## 实际验证摘要

默认 `64 × 256 × 256` 样本已在本执行环境完成生成、校正、解混、补偿、
误差汇总和全量闭合校验：

- 端元条件数：`2.1126982722776826`，低于告警阈值 `30`。
- 漂移估计残差：最大 `0.00666 px`，RMS `0.00190 px`。
- 补偿前平均绝对浓度误差：`0.0193181`。
- 补偿后平均绝对浓度误差：`0.00183564`。
- 改善量：`0.0174824`；四个荧光团全部改善。
- 后补偿 95 分位误差：分别 `0.00215 / 0.00485 / 0.00318 / 0.00721`。
- 误差区间覆盖率：四个荧光团均为 `1.0`。
- 超容差像素比例：四个荧光团均为 `0`。
- 全量闭合校验：`verified=true`，无首分歧。
- 自动化测试：`25` 项全部通过。
- 当前样本、运行缓存和快照总计约 `80 MB`，低于预留的约 `200 MB`。

完整哈希、逐带漂移和逐荧光团指标见
`artifacts/demo/run/diagnostic_report.json` 与 `artifacts/demo/VALIDATION.md`。

## 验证边界

本次实际执行环境为 Linux/Python 3.11.0rc1，PyQt5 使用 `offscreen` 完成渲染烟测；
代码只使用 `pathlib` 和跨平台 API，并提供 PowerShell 脚本与 Windows 命令。
未在本机验证真实 Windows 桌面显示效果本身。确定性声明针对同一
Python/NumPy/LAPACK 运行环境；生成器不使用 BLAS，跨平台逐字节生成有测试覆盖，
NNLS 的极小浮点末位差异由闭合容差 `rtol=1e-6, atol=1e-7` 约束。
