# 旋转机械振动信号无监督异常检测管线 — 技术说明

## 1. 目标与范围

本管线面向**本地确定性合成**的旋转机械振动信号，实现从波形生成、分析窗切分、
短时特征、跨窗特征谱、三种确定性异常检测、定点量化与阈值判定，到内容寻址缓存、
精确失效传导、并发调度、特征点预算与阶段快照恢复的完整数据处理与验证工具。

全部随机源仅出现在样本合成器中，且来自显式种子派生的
`numpy.random.Generator`；特征、谱、检测、量化、调度与缓存代码不构造任何随机源，
因此相同输入与配置在任意进程、任意并发度下逐字段一致。

## 2. 工程结构

```
vibdet/
  errors.py        稳定错误码与失败清单条目
  hashing.py       规范化 JSON 与 SHA-256 指纹
  config.py        YAML 配置、点路径覆盖、参数校验、配置哈希
  synth.py         确定性分段波形生成、缺失填充、重采样/冲击/非有限数处理
  windows.py       分析窗切分与逐窗短时特征、窗指纹
  spectrum.py      跨窗相邻差分、滑窗统计、依赖窗口集合与边界补齐
  detectors.py     稳健马氏距离、谱残差、谱峰值因子（共用同一特征谱）
  quantization.py  归一化、定点量化、固定顺序融合、阈值与边界成因
  cache.py         内容寻址缓存（精确失效判定）
  scheduler.py     有界线程池、稳定排序键、特征点预算、退出信号
  store.py         阶段临时目录写入、整体替换、版本化清单
  segments.py      谱位置判定回映到分段
  edits.py         配置编辑、单窗波形替换、单段删除
  pipeline.py      端到端编排
  cli.py / __main__.py   命令行入口
tools/
  generate_sample.py  生成 config.yaml 与 descriptions.yaml
  acceptance.py       端到端验收脚本
tests/test_pipeline.py  20 个单元/集成测试
```

## 3. 端到端数据流

1. **配置与描述加载**：`config.py` 合并默认配置、本地 YAML 与命令行点路径覆盖；
   `config_hash` 排除运行期调度参数（worker、预算、串行标志），保证续算改预算不被
   误判为配置变更。
2. **确定性波形生成**：每台设备由若干分段拼接；分段内叠加谐波、种子噪声、指数衰减
   冲击与可选故障宽带噪声；缺失采样先置 NaN 再按 `linear/zero/mean` 填充；重采样
   标记点用左右邻点均值替换；填充后仍存在非有限数则抛出 `E_SAMPLE_NONFINITE`。
3. **分析窗与短时特征**：窗起点 `k*hop`，仅保留完整窗。逐窗计算均方根、峭度
   （四阶标准化矩减 3）、峰值，以及基于 `numpy.fft.rfft` 的频带能量；频带边界取自
   配置，频率归一化到奈奎斯特频率，累加严格按采样序号与频带序号升序，窗间不共享
   中间量。窗指纹由波形内容指纹、窗参数指纹与窗序号组合派生。
4. **跨窗特征谱**：对每个特征通道计算 `order` 阶相邻窗差分（左边界用最早真实值
   常量补齐并标记），再在 `stat_window` 个差分上计算均值、离散度（标准差）与极值，
   形成与分析窗等数量的谱位置。每个谱位置显式登记其真实依赖窗口序号集合与跨段
   分段集合；左边界不足时按规则补齐并记录 `padding_count`。
5. **三算法检测**：共用同一谱矩阵。
   - 稳健马氏距离：通道中位数与 `1.4826*MAD` 稳健尺度（含 epsilon 兜底），标准化后
     欧氏距离乘 `scale_multiplier`。
   - 谱残差：通道值与设备长期基线（中位数，含 floor）之比的通道均值，取相对 1 的
     偏差再取 `power` 次幂。
   - 谱峰值因子：通道极值与稳健离散度之比，按位置汇总并加 epsilon。
   检测缓存键同时包含谱位置键、设备级谱统计指纹（中位数/稳健尺度/基线）与算法参数
   指纹，避免设备级统计变化时的错误命中。
6. **归一化、定点量化与判定**：支持 `minmax/robust/maxabs`；归一化值按
   `2**frac_bits` 转为定点整数（舍入或截断两种模式）；融合权重按固定算法顺序
   `(maha, residual, crest)` 在定点整数域逐项加权、半向上舍入，再线性映射回打分
   区间并裁剪。总分恰等于阈值时按 `tie_rule` 决断并标记边界；同时输出高一位宽
   （`frac_bits + ref_extra_bit`）参考总分与差值，两档位判定不一致时区分
   `quantization_truncation` 与 `rounding` 两类成因。
7. **分段回映**：依据每个谱位置登记的跨段集合，把判定回映到分段，输出每段命中的
   谱位置、异常位置与边界位置。

## 4. 指纹、缓存键与精确失效

各层缓存键均为显式内容的 SHA-256，不使用 Python 的 `hash()`/`id()`：

| 层 | 键的输入 |
| --- | --- |
| 窗 | 窗波形内容指纹 + 窗参数指纹 + 窗序号 |
| 谱位置 | 有序依赖窗指纹集合 + 谱参数指纹 + 设备/位置 |
| 算法得分 | 谱位置键 + 设备级谱统计指纹 + 算法参数指纹 |
| 判定 | 三算法该位置得分键集合 + 归一化/量化/融合/阈值参数指纹 |

单窗数据变更只改变其窗指纹，进而只使**依赖窗口集合包含该窗**的谱位置键失效，
其下游算法得分与判定随之失效；其余谱位置在缓存中命中并直接复用，不会重新计算。

## 5. 并发与确定性

- 任务携带稳定排序键 `阶段|设备|序号`，提交前按键排序，结果回收后再次按键排序，
  因而串行与线程池并发的设备、窗、谱位置、算法结果与失败清单顺序逐字段相同。
- 线程池在途任务数严格不超过 `scheduler.max_workers`；`serial=true` 或
  `max_workers=1` 走纯串行分支。
- 预算消费在锁内完成；调度器按设备标识与谱位置序号的稳定顺序消耗特征点，超额位置
  进入持久化待处理队列并记录 `E_POINT_BUDGET_PENDING`，续算时优先消费持久队列。

## 6. 阶段快照与恢复

- 谱构造、算法打分、判定三阶段各自先写入 `tmp/<stage>`，再 `os.replace` 整体替换到
  `snapshots/<stage>/v<N>`。
- 提交前比较临时产物内容指纹与已提交版本：内容不变保留旧版本，内容变化才递增版本；
  清单 `manifest.json` 记录全局版本、配置哈希、波形指纹、已消耗预算、待处理队列与
  各阶段版本/指纹/路径，并以原子替换写盘。
- 收到 SIGINT/SIGTERM 后，管线在设备/谱位置检查点完成当前阶段并输出
  `output/diagnostics.json`；被 SIGKILL 强制终止后，续算只信任已提交快照与清单中的
  待处理队列。

## 7. 错误码

| 错误码 | 触发条件 |
| --- | --- |
| `E_DEVICE_UNKNOWN` | 设备不在波形描述中或缺少分段定义 |
| `E_WINDOW_INVALID` | 窗长/窗移非正、窗移大于窗长、波形短于窗长等 |
| `E_SPECTRUM_TOO_SHORT` | 窗数不足以构造二阶及以上特征谱 |
| `E_SAMPLE_NONFINITE` | 缺失填充后仍存在非有限采样 |
| `E_POINT_BUDGET_PENDING` | 特征点预算不足，谱位置进入待处理队列 |

## 8. 命令行用法

```bash
pip install -r requirements.txt
python -c "import numpy, yaml"

# 生成本地确定性样本
python -m vibdet init --workspace ws
# 全量运行（可重复点路径覆盖任意配置项）
python -m vibdet run --workspace ws \
  --override window.length=200 --override window.hop=100 \
  --override spectrum.order=3 --override quantization.frac_bits=10
# 编辑：单窗替换 / 单段删除 / 配置项修改
python -m vibdet edit --workspace ws --kind replace-window \
  --device DEV-A --index 2 --value 3.0
python -m vibdet edit --workspace ws --kind delete-segment \
  --device DEV-A --segment-id A-seg2
python -m vibdet edit --workspace ws --kind config \
  --path detectors.maha.scale_multiplier --value 1.8
# 端到端验收
python tools/acceptance.py --workspace artifacts/acceptance_ws
```

## 9. 验证结果（本机实测）

- 单元/集成测试：`python -m pytest tests/ -q` → **20 passed**。
- 验收脚本：`python tools/acceptance.py --workspace artifacts/acceptance_ws`
  → **18/18 检查通过**，覆盖跨窗特征谱与补齐、跨段依赖登记、三算法定点得分、
  单窗精确失效、完整编辑序列后增量与空白全量逐字段一致、串行/并发一致、
  预算待处理与续算、阶段快照版本清单。
- 信号演练：连续 6 次 SIGTERM 均返回码 0 并产出 `shutdown_after_stage` 诊断；
  SIGKILL 后已提交快照与待处理队列完整，续算成功。
- 确定性：相同输入与配置重复运行的 `devices/failures/pending_queue` 规范化哈希完全
  相同；切换任一算法超参数（马氏缩放、残差幂次、峰值因子 epsilon）均真实改变得分。
