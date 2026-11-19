# HDR → SDR 专业色彩转换工作台 — 技术说明

## 1. 范围与确定性原则

工作台在本地用固定种子合成线性光 HDR 帧序列，经参数化颜色管线产出 8 位
SDR 帧与色彩指标，并以版本化操作序列维护状态、支持增量重算、历史重建与
崩溃恢复。**全部数值路径确定性**：

- 固定 `numpy.random.default_rng(seed)`（PCG64）与固定调用顺序；
- 全部像素计算在 `float32` 下执行，显式逐通道乘加（`_matmul_pixel`），
  不依赖可能随 BLAS 后端变化的矩阵乘；
- 规约（均值）统一转 `float64` 成对求和；
- 序列化采用键排序、无空白的规范化 JSON（`canonical_json`）；
- 状态 id 与输出哈希均为 SHA-256，内容寻址，与操作粒度无关。

## 2. 模块结构

| 模块 | 职责 |
| --- | --- |
| `halffloat.py` | binary16 编解码、RNE 舍入、NaN/Inf/饱和口径 |
| `synthesis.py` | 确定性帧序列合成、half 落盘、亮度真值清单 |
| `pipeline.py` | 色域矩阵、曝光、Reinhard/ACES、OETF、8 位量化、指标 |
| `state.py` | `PipelineState`、版本化操作序列、失效闭包、历史重放 |
| `persistence.py` | 原子写、manifest/operations、快照新旧判定、孤儿回收 |
| `engine.py` | 实算编排、增量重算、历史重建、全量校验、恢复 |
| `display.py` | OpenGL 离屏显示，失败回退 numpy 软件缓冲 |
| `shell.py` | 交互式时间轴/曲线/状态/闭包命令行 |
| `cli.py` | `generate / demo / selftest / shell / verify / hashes` |
| `errors.py` | 7 个稳定业务错误码与异常链路 |

## 3. 半浮点口径（要求 1）

- half→float32：binary16 有限值均可被 binary32 精确表示，转换无损；
  舍入模式为 IEEE 754 就近舍入偶数优先（RNE）。
- float32→half：24 位有效数按 guard/round/sticky 做 RNE，subnormal 同口径。
- 防御策略：+/-Inf 饱和为 +/-65504；NaN 映射为 0；超界有限值饱和。
- 标量参考实现与 numpy 向量化实现在 60 万随机样本 + subnormal/Inf/NaN
  边界值上逐位一致（`tests/test_halffloat.py`）。

## 4. 颜色管线数值口径（要求 3）

固定步骤顺序（每步中间结果均为 float32）：

1. 曝光：`x * 2**stops`；
2. 色域：源 BT.2020 → 目标 BT.709/BT.2020 的 3×3 矩阵。矩阵由原色 xy
   与 D65 白点推导 `M = XYZ709^-1 @ XYZ2020`，导入期算一次后固化为
   float32 常量 `M_2020_TO_709`；BT.2020 目标为恒等矩阵；
3. 色调映射（逐通道，负值先归零）：
   - Reinhard（扩展式）：`x*(1 + x/w^2)/(1 + x)`，`w=tonemap_param>0`；
   - ACES（Narkowicz 2015 拟合，常数固定）：
     `(x*(2.51x+0.03))/(x*(2.43x+0.59)+0.14)`；
4. 饱和：固定 `clip([0,1])`；
5. OETF：
   - sRGB：`x<=0.0031308 → 12.92x`，否则 `1.055 x^(1/2.4)-0.055`；
   - BT.709/BT.2020：`x<0.018 → 4.5x`，否则 `1.099 x^0.45-0.099`；
6. 8 位量化：`floor(x*255 + 0.5)` 后 clip 到 `[0,255]`。

亮度系数：BT.709 `(0.2126,0.7152,0.0722)`、BT.2020
`(0.2627,0.6780,0.0593)`，均为固定 float32 常量。

## 5. 状态、闭包与历史重建（要求 2、4）

- 状态字段：`colorspace / exposure_stops / tonemap / tonemap_param / oetf`，
  规范化 JSON 的 SHA-256 即 state id。
- 操作记录含 `version / kind / frames / params / parent_state_ids / closure`，
  父状态链完整可重放（`StateTable.replay_to`）。
- 失效闭包：逐帧比较新旧参数，只有真正变化的帧入闭包；空操作闭包为 0。
- 引擎对闭包内帧实算并原子落地，闭包外帧 state id 与输出哈希保持不变，
  并在每次提交后做 `E_CLOSURE_DIVERGE` 自检。
- 区间操作与等价逐帧操作到达同一状态时，state id 与输出 SHA-256 完全
  一致（版本号属于过程元数据，不影响输出内容）。
- 历史重建 `rebuild_historical(version)` 从操作序列重放后**重新实算**，
  并与该版本留存产物的哈希比对，禁止近似缓存冒充实算。

## 6. 持久化与恢复（要求 5、6）

提交协议：新产物先写 `tmp/` → fsync → `os.replace` 原子改名 → 重写
`operations.jsonl` → 原子替换 `manifest.json`。只有 manifest 替换成功，
该版本戳才算已提交；任何中途崩溃都不会产生部分提交。

- 快照整体导入经 `reject_if_stale` 以 `committed_at_seq` 版本戳判新旧，
  旧快照返回 `E_SNAPSHOT_STALE`，不盲目录入。
- 重启恢复只信任 manifest 的版本戳与内容哈希：
  - `tmp/` 内文件与不符合版本戳命名的产物视为孤儿，回收并写入 `recovery.log`，
    触发 `E_RECOVER_ORPHAN`（引擎捕获后进入可继续操作状态）；
  - 合法历史版本产物保留，供历史状态重建；
  - 哈希不符返回 `E_HASH_MISMATCH`。
- 受控退出额外写 `shutdown_report.json`；强制终止路径不产生该文件，
  恢复时不得伪造退出报告。

## 7. 错误码覆盖（要求 7）

| 错误码 | 触发场景 |
| --- | --- |
| `E_FRAME_FORMAT` | 非法维度、帧字节数不符、zlib 压缩数据损坏 |
| `E_STATE_VERSION` | schema 版本冲突、帧号越界、重放版本越界 |
| `E_HASH_MISMATCH` | 产物回读/全量校验/历史重算哈希不符 |
| `E_CLOSURE_DIVERGE` | 闭包外帧状态被意外改变 |
| `E_SNAPSHOT_STALE` | 导入快照版本戳旧于当前 |
| `E_TONEMAP_PARAM` | 非法色域/算子/OETF/曝光/非正算子参数 |
| `E_RECOVER_ORPHAN` | 恢复时发现并回收未提交孤儿产物 |

所有错误均以 `WorkbenchError(code, message, details)` 抛出，CLI 捕获后
输出结构化 JSON 并返回非零码，工作区保持可继续操作。

## 8. 显示层（要求 8）

`display.create_display()` 先尝试 PyOpenGL 离屏上下文与纹理上传，任何
平台/无头失败都回退到 `SoftwareDisplay`（numpy 确定性最近邻降采样 +
ASCII 时间轴）。显示层不参与哈希与指标，因此 GL/软件差异不影响交付的
可重复性。无头验证环境实际使用软件后端。

## 9. Windows 落地说明

- 仅使用 Python 3.11 标准库 + numpy 1.26.4 + PyOpenGL 3.1.7，无容器、
  无网络依赖、无第三方素材；
- 原子写使用 `os.replace`（Windows 上为原子替换语义），路径全部经
  `os.path.join` 构造；
- GL 离屏在无 GPU/无窗口系统机器上自动软件回退；
- 推荐命令：
  `python -m pip install -r requirements.txt`
  `python -m hdr_sdr_workbench --workspace ws demo --clean`
  `python -m pytest`
