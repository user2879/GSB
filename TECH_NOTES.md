# 说话人日志组件技术说明

面向 16kHz / 单声道 / 16bit PCM 长音频的可集成说话人日志（speaker diarization）
组件。仅依赖 `numpy`，Tkinter 为标准库可选依赖；无网络、无外部服务、无容器。

## 1. 目录结构

```
src/diariz/
  config.py        配置数据类、严格校验、配置/投影指纹
  audioio.py       手工 RIFF/WAV 解析、严格格式校验、确定性写入与截断注入
  synth.py         固定种子源-滤波器多说话人合成（停顿 + 重叠）
  features.py      重叠分窗、帧级能量/基频/倒谱、确定性投影嵌入（float64）
  quantize.py      定点量化、逐维残差、纯整数 Q16 归一化相似度
  clustering.py    字典序增量聚类、等阈值边界窗、簇合并与代表整合
  segmentation.py  重叠帧余量仲裁、三种边界规则、最短段/静音后处理
  cache.py         内容指纹缓存键、原子持久化、复用/重算清单
  concurrency.py   线程池、稳定排序键、预算扣减与串行等价
  snapshot.py      临时目录暂存 + 整体替换、版本号与冲突识别
  errorprop.py     两档位四环节误差传播与边界翻转归因
  faults.py        四类真实故障注入与单音频隔离
  pipeline.py      端到端编排、单窗替换精确失效
  visualize.py     Tkinter 时间线（缺失时降级为 HTML/SVG）
demo.py            一键完整示例（合成→全流程→复用→归因→故障→快照）
run_diariz.py      命令行入口（synth / diarize / visualize）
tests/             17 个 unittest 用例
```

## 2. 数据流与关键设计

1. **加载校验**：`audioio.read_wav` 手工遍历 RIFF 块，校验 RIFF/WAVE 头、
   PCM 格式标签、16000Hz、单声道、16bit、数据块长度与采样数一致、采样值有限；
   不合规抛 `E_AUDIO_FORMAT`。配置由 `config.parse_config` 校验，非法抛
   `E_CONFIG_INVALID` 并在任何产物写入之前终止。
2. **重叠分窗与嵌入**：`features.plan_windows` 按窗长/窗移生成等间隔重叠窗，
   记录起止采样、覆盖帧集合与窗间重叠关系。帧特征含短时能量、自相关基频与
   12 阶倒谱；窗内按帧序号升序累加。投影矩阵由配置指纹派生，前若干维恒等对齐
   以保留 f0/共振峰判别力，其余维为种子化随机混合；全程 float64，不使用全局
   随机状态，两次运行逐元素一致。
3. **定点量化**：`quantize.quantize_embedding` 以 `2^-frac_bits` 为步长，按
   字长（8/16/32）裁剪，逐维记录整数、反量化值与残差（浮点值−反量化值）及
   最大残差维。切换字长/小数位宽会真实改变整数与残差。
4. **整数相似度与聚类**：`integer_similarity` 用任意精度整数按维升序计算
   点积/范数，输出 Q16 定点余弦，阈值同样量化为 Q16 后整数比较；相似度恰等
   阈值判同并记为边界窗。新窗按簇 ID 字典序取首个达阈值簇，否则新建；整合
   阶段合并代表仍达阈值的簇，保留较小 ID，成员按窗序号升序追加。
5. **帧级分段与仲裁**：重叠帧取覆盖窗中相似度余量（sim−threshold，Q16）
   最大者；余量相等按窗序号升序。支持 `frame_edge` / `window_midpoint` /
   `margin_weighted` 三种边界规则，再做最短分段丢弃与过近同说话人合并。
6. **内容指纹与精确失效**：缓存键分层为
   `embed(内容+投影指纹) → quant(嵌入键+字长+小数位) → compare(两量化键+阈值)
   → segment(比较键集合+配置)`。单窗替换仅使该窗内容及其重叠比较、段边界
   失效；改阈值复用嵌入/量化、重算比较；换位宽复用嵌入、重算量化与下游。
7. **两档位误差归因**：`errorprop.propagate` 以配置档位与高一档位完整计算
   量化/相似度/聚类/分段四环节差值，独立判定各窗首个超差环节；说话人数或
   边界不一致时区分量化截断、阈值邻域、边界仲裁三类归因并给出判据数值。
8. **快照/并发/故障**：线程池并发执行嵌入与比较，任务携带
   `文件名|窗序号|阶段` 稳定排序键，提交与预算扣减按规范序进行，与串行结果
   逐字段一致。快照先写临时目录再整体替换，清单含全局版本、配置哈希、音频
   游标与产物版本，提交前检测版本冲突。故障注入 INJ-01..04 覆盖非有限采样、
   全静音、阈值越界、截断数据块，单音频失败不影响其余音频。

## 3. 缓存键与失效范围

| 变更 | embed | quant | compare | segment |
| --- | --- | --- | --- | --- |
| 单窗音频替换 | 仅受影响内容键重算 | 受影响窗重算 | 引用该窗的比较重算 | 重算 |
| 仅改阈值 | 全复用 | 全复用 | 全重算 | 重算 |
| 改字长/小数位 | 全复用 | 全重算 | 全重算 | 重算 |
| 改边界/最短段规则 | 全复用 | 全复用 | 全复用 | 重算 |

持久化缓存为内容寻址（相同内容的多个窗共享一份载荷），窗序号在调用方重新
绑定，因此去重不会污染窗归属。每次运行的复用/重算清单写入结果 JSON 的
`reuse_report`，可逐键核对。

## 4. Windows 本地运行

```powershell
python --version                 # 需 3.10+
python -m pip install numpy
python demo.py                   # 端到端示例，生成 samples/ 与 output/
python -m unittest discover -s tests -v
python run_diariz.py synth --input samples
python run_diariz.py diarize --wav samples/synthetic_3spk.wav `
    --config samples/diar_config.json --output output/run
python run_diariz.py visualize --summary output/summary.json `
    --html output/timeline.html
```

Tkinter 在官方 Windows/macOS 安装包中随 Python 自带；极简 Linux 环境若缺少
原生 Tk 运行时，可视化自动降级为自包含 HTML/SVG，不影响主流程。

## 5. 实际验证结果（本机 Python 3.11 + numpy 1.26）

- `python demo.py`：59 个重叠窗、3 个说话人簇、7 个分段；确定性校验
  窗偏移与嵌入两次运行完全一致；INJ-01..04 四类故障全部命中。
- `python -m unittest`：17/17 通过，覆盖格式与配置校验、确定性、量化残差、
  等阈值边界、聚类字典序、分段仲裁、精确失效、两档位归因、快照冲突、
  故障隔离与串并发等价。
- 单窗替换：嵌入 56 复用 / 3 重算，比较 1553 复用 / 158 重算，分段重算 1。
- 阈值修改：嵌入与量化全部复用，比较与分段重算。
- 位宽修改（16bit/10frac → 32bit/12frac）：嵌入全部复用，量化与下游重算；
  56 窗四环节差值在容差内，3 窗边界受仲裁规则影响，归因为
  `boundary_arbitration_rule`，说话人数两档位一致（均为 3）。

## 6. 已知边界

- 声纹嵌入为轻量声学特征（能量/基频/倒谱 + 确定性投影），并非神经网络
  x-vector；目标是可复现、可定点化与可解释，而非产业级识别精度。
- 重叠发声区不单独输出重叠说话人标签，重叠帧通过余量仲裁归属单一说话人。
