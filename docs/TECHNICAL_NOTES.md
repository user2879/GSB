# 透明介质折射合成组件 — 技术说明

## 1. 范围与约束

- 纯 Python 3.11 + NumPy 实现，无 OpenGL/上下文、无外部模型或线上服务。
- 全部输入由固定种子（默认 `20261119`）程序化生成，RNG 为纯 Python SplitMix64
  （`refrac/rng.py`），与 numpy 版本无关，保证跨环境逐字节一致。
- 异常注入只通过场景 JSON 与注入配置（参数层）生效，不修改源码、不篡改验收产物。

## 2. 模块结构

| 模块 | 职责 |
| --- | --- |
| `refrac/rng.py` | 确定性 SplitMix64 随机源 |
| `refrac/vectors.py` | 向量运算、零向量退化归类（E_RAY_DEGENERATE） |
| `refrac/geometry.py` | 光线、球体/平板/背景平面求交，退化相交归类 |
| `refrac/optics.py` | 斯涅尔折射、临界角、全反射判定、折射率范围校验 |
| `refrac/absorption.py` | 比尔—朗伯衰减、采样积分与解析式对照、负吸收校验 |
| `refrac/scene.py` | 场景数据结构、确定性生成器、规范 JSON 序列化 |
| `refrac/camera.py` | 针孔相机主光线、解析投影覆盖真值 |
| `refrac/injection.py` | 注入配置 schema 校验与场景参数覆盖 |
| `refrac/inject_cases.py` | 标准异常注入用例集 |
| `refrac/tracer.py` | 递归深度/TIR 预算控制、折射合成、逐像素隔离 |
| `refrac/diagnostics.py` | 路径报告、首个受影响像素、最小反例与复算验证 |
| `refrac/persistence.py` | float32 `.npy` 落盘与 SHA-256 哈希 |
| `refrac/cli.py` | `clean/generate/render/suite/replay` 命令行 |

## 3. 关键物理与数值约定

- 光线方向为单位向量；法线在追踪器内统一翻转到“朝向入射介质一侧”。
- 折射使用向量形式斯涅尔定律，`sin²θ_t = η²(1-cos²θ_i)`：
  - `> 1` 判定为全反射（TIR），按镜面反射改向继续传播，光线不丢弃；
  - 临界角 `arcsin(n_t/n_i)` 仅在光密→光疏时存在，否则归类 `NO_CRITICAL_ANGLE`。
- 吸收：`T = exp(-Σ σ·(L/S))`，固定顺序累加光学厚度；常量 σ 下与解析式
  `exp(-σL)` 的偏差 ≤ 1e-10（实际约 1e-16）。
- 两类独立预算：
  - `max_depth`：折射界面穿越上限，超限归类 `E_DEPTH_BOUNDS` 并隔离该像素；
  - `max_tir_bounces`：TIR 弹跳上限，防止掠射陷获光线无限弹跳，
    超限以 `depth_limit` 终止并保留累计颜色（不报错、不丢光线）。
- 输出为线性空间 float32 RGBA；隔离像素写入品红哨兵 `(1,0,1,1)`。

## 4. 业务错误码

| 错误码 | 触发条件 | 处置层级 |
| --- | --- | --- |
| `E_IOR_RANGE` | ior ≤ 0 或超出声明范围 [1.0, 3.0] | 像素隔离 |
| `E_RAY_DEGENERATE` | 光线方向向量为零 | 像素隔离 |
| `E_DEPTH_BOUNDS` | 深度参数 <1 或触及折射深度上限 | 像素/运行级 |
| `E_ABSORB_NEGATIVE` | 吸收系数含负分量 | 像素隔离 |
| `E_INTERSECT_DEGENERATE` | 半径/厚度非正等退化相交 | 像素隔离 |
| `E_PIXEL_ISOLATED` | 单像素失败的帧级归类码 | 帧级 |
| `E_INJECT_CONFIG` | 注入配置 schema 非法/指向未知体 | 运行级 |

单像素异常在 `render_frame` 内被捕获并写入隔离记录，整帧继续渲染；
遍历顺序固定为行优先、列升序，首个受影响像素据此确定。

## 5. 最小反例

- 反例为可序列化 JSON 片段，含 `code / pixel / required_fields / trigger_fields / fragment`。
- 验证规则（`verify_counterexample_minimality`）：
  - 完整片段单独 `replay` 复现同一错误码；
  - 移除任一触发字段（回退注入覆盖）或改变像素坐标后不再复现。

## 6. 复现命令

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m refrac --workspace . suite
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Linux/macOS/WSL：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m refrac --workspace . suite
.venv/bin/python -m unittest discover -s tests -v
```

单像素复算示例：

```bash
.venv/bin/python -m refrac --workspace . replay \
  --scene samples/scene.json --injection samples/inj_zero_direction.json --row 0 --col 0
```

## 7. 产物目录

- `samples/`：场景 JSON、元数据清单（含法线真值与投影覆盖像素数）、注入配置。
- `artifacts/`：各次运行的 float32 RGBA `.npy`。
- `reports/`：逐像素路径报告、运行摘要（含哈希、隔离记录、最小反例）、`run_manifest.json`。

## 8. 实际验证结果（本地）

- 场景两次生成、同场景两次合成均逐字节一致（SHA-256 相同）。
- 吸收采样积分与解析式最大偏差 ~1.1e-16，容差 1e-10 通过。
- 采样数 8→16：差异像素全部属于吸收像素集合（常量 σ 下实测零差异）。
- 深度上限 10→2：仅触及上限的像素终止原因变化，其余像素逐字段不变。
- TIR：基线场景与 `tests/probe_scenes.py` 探针均观测到带位置与次数标注的全反射事件，
  临界角与声明 ior 一致；ior=1.0 退化场景归类 `NO_CRITICAL_ANGLE`、零隔离。
- 注入套件 9 个用例全部得到预期归类；像素级注入只影响目标像素，最小反例全部满足最小性。
- `unittest` 40 个用例全部通过。
