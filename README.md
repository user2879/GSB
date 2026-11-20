# HDR → SDR 可集成色彩转换领域组件

纯 Python 3.11 + numpy 实现的确定性 HDR→SDR 色彩转换组件与完整示例。无 GPU、无编解码器、
无容器、无任何在线依赖；半浮点处理仅使用标准库与 numpy 的 `float16`。

## 1. 环境与安装（Windows / 本地）

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

依赖固定为 `numpy==1.26.4`、`pytest==8.0.2`（Linux/macOS 同样适用）。

## 2. 示例命令

所有产物都写入工作区内的本地路径；运行入口会先清空目标工作区，保证可重复。

```bat
:: 1) 生成 64 帧 960x540 fp16 确定性序列与真值清单
python -m hdr_sdr_pipeline.runner generate --work-dir work\demo

:: 2) 全量构建 + 增量重算 + 注入检测 + 首像素定位 + 最小反例 + 正常关闭
python -m hdr_sdr_pipeline.runner run --work-dir work\demo

:: 3) 强制终止语义与重启恢复（孤儿回收、恢复后与全量重算一致）
python -m hdr_sdr_pipeline.runner crash-demo --work-dir work\crash

:: 测试
python -m pytest
```

`run` 会额外产生两个对照工作区：`work/demo_cmp`（逐帧路由对照）、
`work/demo_fault`（单状态故障工作区）、`work/demo_const`（色调映射常量篡改对照）。

## 3. 模块结构

| 模块 | 职责 |
| --- | --- |
| `synth.py` | 固定种子帧序列生成、fp16 落盘、真值清单（峰值/均值/采样点/区域） |
| `config.py` | 带版本号的 `ProcessingState`、管线阶段、固定常数、参数合法域 |
| `colorspace.py` | BT.709/BT.2020 固定矩阵与 luma 系数，显式 float32 运算顺序 |
| `tonemap.py` | Reinhard / ACES 拟合曲线、钳制与高光饱和、域错误 |
| `transfer.py` | sRGB / gamma 输出传递函数 |
| `quantize.py` | 8 位量化（步长 1/255）、抖动开关（关闭即确定性） |
| `pipeline.py` | 逐帧七阶段管线、阶段 float32 输出与内容哈希、注入钩子 |
| `ledger.py` | 账本结构：状态历史、阶段哈希、操作日志、指标 |
| `engine.py` | 增量引擎：失效闭包、逐帧/区间路由、闭包核对、指标重算 |
| `metrics.py` | 仅凭 8 位输出 + 真值清单独立复算的质量指标 |
| `injection.py` | 可序列化的规格层/管线层注入配置 |
| `detector.py` | 规格校验、阶段哈希检测、已检出/漏检/误报三分类 |
| `locate.py` | 首个受影响像素定位与最小反例（delta debugging） |
| `persistence.py` | 临时文件+原子改名、孤儿回收、关闭标记、陈旧快照 |
| `errors.py` | 八个固定业务错误码与类型化异常 |

## 4. 关键数值口径（冻结）

- **半浮点**：生成全程 float32，落盘前 `astype(float16)`，采用 IEEE 754 binary16
  就近偶舍入（round-to-nearest-even），以小端 `<f2` 存储；fp16→fp32 为精确扩展。
- **色彩空间**：合成帧以 BT.709 linear 创作；目标 BT.709 为恒等矩阵，BT.2020 使用固定
  BT.709→BT.2020 线性矩阵，逐通道按行序做 float32 乘加（不调用 BLAS，跨平台逐字节一致）。
- **曝光**：线性域乘以 `2**ev`（float32）。
- **Reinhard**：`x(1+x/W²)/(1+x)`；**ACES**：Narkowicz 拟合
  `(x(Ax+B))/(x(Cx+D)+E)`，常数 A=2.51、B=0.03、C=2.43、D=0.59、E=0.14。
- **域处理**：非有限值 → `E_TONEMAP_DOMAIN`；负值钳到 0；超过 64.0 的 HDR 溢出钳到
  64.0（确定性高光饱和），算子输出再钳到 [0,1]。
- **输出编码**：sRGB 线性段阈值 0.0031308；gamma 合法域 [0.1, 10.0]，越界 →
  `E_INJECT_BOUNDS`。
- **量化**：步长 `1/255`，默认关闭抖动，`rint` 就近偶舍入后转 uint8。
- **哈希**：SHA-256，输入为 dtype+shape+ C 序原始字节；JSON 用排序键的规范形式。

## 5. 增量闭包与独立复算

- 参数变更映射到其首个生效阶段（`PARAM_ENTRY_STAGE`），闭包 = 该阶段及其全部下游；
  引擎断言实际重算集合（帧、阶段、区间）与预测闭包完全一致，否则
  `E_CLOSURE_DIVERGE`，禁止整段重建。
- 闭包外帧的 8 位输出哈希保持不变；逐帧路由与区间路由经由同一 `process_frame`
  原语，到达同一最终状态时输出哈希逐字节相同（示例与测试均有断言）。
- 指标只读取 8 位输出帧与 `truth.json`，配合状态参数解析参考值，不引用任何管线中间量；
  同输出两次计算的规范 JSON 完全一致。区域内等权像素、区间内等权帧、float64 归约。

## 6. 故障注入、检测与定位

- 注入配置为可序列化 JSON，随账本记录；注入只能经输入/参数/注入上下文生效。
- 规格层：峰值越界、传递函数非法、采样真值篡改、色调映射常量篡改；
  管线层：指定帧指定阶段的中间数组替换。
- 检测报告区分 **已检出 / 漏检 / 误报**，漏检项附判定依据（如瞬时故障在提交前被干净
  重算覆盖，账本无任何可观测痕迹）。
- 定位顺序固定：帧号升序 → 行优先 → 列升序；报告首个受影响像素的坐标、错误码、
  观测值/干净值、误差区间与触发注入片段。最小反例按固定序做元素消除，移除任一剩余
  元素即不再复现同一错误码。

## 7. 业务错误码

| 错误码 | 触发场景 |
| --- | --- |
| `E_FRAME_FORMAT` | 帧字节长度/形状/半浮点解码非法、注入替换形状不符 |
| `E_STATE_VERSION` | 状态/账本 schema 版本或字段不受支持 |
| `E_HASH_MISMATCH` | 真值清单、阶段或输出内容哈希与重算不一致 |
| `E_CLOSURE_DIVERGE` | 实际重算集合与失效闭包不符 |
| `E_INJECT_BOUNDS` | 峰值/gamma/白平衡等参数超出合法域 |
| `E_DETECT_MISS` | 应检出的注入未被任何检测器命中 |
| `E_SNAPSHOT_STALE` | 恢复快照早于最近一次已提交账本 |
| `E_TONEMAP_DOMAIN` | 色调映射输入非有限值 |

## 8. 持久化与终止语义

- 所有文件先写唯一 `*.tmp.<pid>` 并 `fsync`，再以 `os.replace` 原子改名（NTFS/POSIX
  均为原子）；账本总是在其引用的帧文件之后提交，写未完成即视为未提交。
- 重启只依据最近一次已提交账本恢复；未引用文件与残留临时文件作为孤儿移入 `orphans/`
  并写入 `recovery_log.json`；正常退出写 `CLOSED` 关闭标记，恢复时旧标记失效。
- 外部快照早于已提交序列号或哈希不符时抛 `E_SNAPSHOT_STALE`。

## 9. 已记录的实际验证结果

- `python -m pytest`：**44 个测试全部通过**。
- 全尺寸 `run`：闭包外帧哈希不变、逐帧/区间路由输出一致；注入 6 项 → 检出 5、漏检 1、
  另含 1 个误报演示；首像素定位到帧 20 的 (row=0, col=0)，最小反例仅保留
  `inj_stage_tm`。
- `crash-demo`：回收 1 个临时孤儿，恢复序列号 2，恢复后继续重算与未中断全量运行的
  输出哈希完全一致。
- 两次独立生成：64 个 fp16 帧与 `truth.json` 逐字节相同；两次相同管线运行的
  `ledger.json` 与全部输出哈希逐字节相同。
