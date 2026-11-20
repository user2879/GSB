# 确定性振动分段异常检测与增量验证工具

本项目在本地生成确定性单通道设备振动波形，并按固定段长切片，执行“特征提取 -> 三算法分段打分 -> 归一化/定点化/阈值判定 -> 汇总”四阶段无监督异常检测。核心能力是使用分层内容指纹与依赖指纹进行精确缓存失效：追加、替换或删除分段，调整算法超参数、频带、量化档位或阈值时，只重算真正受影响的分段、算法与下游判定。

项目不依赖容器、数据库或网络数据；运行时仅使用本地 YAML、Python 3.11+、NumPy 与 PyYAML。

## 快速开始（Windows）

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 -m unittest discover -s tests -v
py -3.11 tools\verify.py
```

运行基础管线：

```powershell
py -3.11 -m vibration_pipeline run --mode incremental `
  --config config\pipeline.yaml `
  --data data\devices.yaml `
  --snapshot-dir artifacts\snapshot `
  --report artifacts\run_report.json
```

运行带四类真实注入故障的完整报告：

```powershell
py -3.11 -m vibration_pipeline run --mode full `
  --injections config\injections.yaml `
  --snapshot-dir artifacts\injection_snapshot `
  --report artifacts\injection_report.json
```

应用示例编辑覆盖：

```powershell
py -3.11 -m vibration_pipeline run --mode incremental `
  --edits data\edits_demo.yaml `
  --snapshot-dir artifacts\snapshot `
  --report artifacts\edited_report.json
```

同一 `--snapshot-dir` 连续运行即为增量模式；`--mode full` 不使用上轮缓存。CLI 正常完成、包含业务失败或收到可处理退出信号时均返回整数退出码 `0`；文件或参数级致命错误返回 `2`。

## 文件布局

```text
config/pipeline.yaml        # 管线配置
config/injections.yaml      # 独立注入配置
data/devices.yaml           # 确定性波形设备描述
data/edits_demo.yaml        # 追加/替换/删除覆盖样例
vibration_pipeline/         # Python 包
  config.py                 # 配置解析、候选与业务码校验
  data.py                   # 波形生成、切片、预处理、编辑
  features.py               # 时域与频带特征、内容指纹
  algorithms.py             # 三种确定性评分算法及依赖上下文
  quantization.py           # 归一化、定点化、融合与阈值决断
  snapshot.py               # 临时目录写入与阶段目录整体替换
  engine.py                 # 四阶段、线程池、缓存失效、报告
  cli.py                    # 命令行与退出信号
tests/test_pipeline.py      # 9 项自动化验收测试
tools/verify.py             # 端到端增量/全量/注入/并发验证器
docs/RESULTS.md             # 最近一次真实验证结果
artifacts/report.json       # 带注入的完整 JSON 报告
artifacts/incremental_report.json
artifacts/verification_report.json
```

## 确定性波形与分段

每台设备由固定 seed、采样率、基频转速、转速漂移、多阶谐波、冲击脉冲、底噪与异常幅度系数生成。波形使用绝对采样时刻解析计算，因此单段替换或追加不会改变其他段采样值。异常段数量由 `anomaly_ratio` 决定，并从设备固定随机序列中确定性选取。

生成后按 `segment_length` 切片。每段保留：

- `segment_index` 与 `start_sample/end_sample`；
- 原始波形、缺失采样索引、重采样标记索引；
- 缺失点与重采样点经段内线性插值后的处理波形；
- 波形内容指纹与预处理指纹。

设备描述也支持：

```yaml
missing_indices: [12, 91]     # 全局采样索引
missing_stride: 191           # 自动生成全局缺失点
resampled_indices: [33]
resampled_stride: 263
```

`segment_edits` 或 `--edits` 覆盖文件支持：

```yaml
segment_edits:
  pump-a:
    append:
      - count: 1
        seed_delta: 9001
        source: demo-append
    replace:
      - segment_index: 2
        seed_delta: 707
    delete: [4]
```

`replace.samples` 也可提供显式单段采样数组；长度必须等于段长。

## 特征、算法与量化

特征模式为 `time_domain`、`band_energy` 或 `combined`。固定顺序为：

1. 段内采样序号升序；
2. 时域特征 `rms, kurtosis, peak`；
3. 频带序号升序的 DFT 能量。

频带能量由 `numpy.fft.rfft` 的确定性离散频谱计算，范围采用下界包含、上界不包含，最后一个频带包含 Nyquist。所有特征值稳定到 12 位小数。

三种算法使用算法名字典序：

- `crest_factor`：各段 `peak/rms` 在尾部窗口内取均值。
- `robust_mahalanobis`：设备内各特征中位数与 `1.4826 * MAD` 稳健缩放，构造稳健协方差矩阵并计算马氏距离；秩亏时使用固定 `rcond` 的 Hermitian 伪逆。
- `spectral_residual`：各频带能量与设备长期频带基线之比，再取固定顺序均值。

每个算法可配置多个超参数候选并由 `candidate_index` 选定。三算法原始分数均使用固定 NumPy/Python 运算顺序。

阈值阶段先按算法 `normalization` 归一化，再按：

```text
fixed integer ticks = round_half_away(normalized * 10**decimal_places / quantum) * quantum
```

转为定点整数。随后按算法名字典序和融合权重合成总异常分，裁剪到 `score_range`，以同样规则量化。`score_ticks == threshold_ticks` 时记为边界分段，并由固定 `tie_policy` 决定异常标记。边界清单携带分数、阈值、算法参考项、量化指纹和判定依据。

默认样本已设置到量化敏感阈值：把 `quantum` 从 `1` 改为 `2` 会同时改变定点整数及至少一个异常标记。

## 缓存键与精确失效

缓存不是按文件名或运行时间失效，而是按稳定指纹：

- 分段波形键：设备、段序号、起始采样、采样率、原始采样字节指纹、缺失/重采样标记、数据来源。
- 时域/频带特征键：波形键、预处理指纹、段长、特征公式；频带键另含采样率与频带边界。
- 特征内容指纹：波形内容、预处理、段长、频带配置、模式、固定顺序特征值。
- 原始算法分键：波形与预处理指纹、段序号、算法参数哈希、算法引用特征值/特征键、算法上下文指纹。
- 定点算法分键：原始分键、归一化配置哈希、量化档位与小数位宽。
- 最终判定键：各算法定点分键、融合权重、分数区间、阈值整数、量化配置与 tie policy。

依赖上下文单独记录：

- 稳健马氏距离依赖设备内全部参与模型的成功分段；
- 谱残差依赖设备内长期频带基线所用的全部成功分段与频带特征；
- 峰值因子只依赖尾窗内分段的时域键；
- 最终判定只依赖成功算法的定点分与阈值配置。

因此频带变化会重算频带特征、马氏距离和谱残差，但时域特征与峰值因子原始分可直接复用；单个算法候选参数变化只重算该算法及其定点分和判定；量化变化复用全部原始算法分；阈值变化只重算最终判定。

每次运行输出逐段 `reuse` 与 `recompute` 清单，包含层名、算法、缓存键与依据。清单按 `device_id -> segment_index -> algorithm -> layer` 的稳定键排序。

## 并发、快照与退出

线程池最大 worker 数严格取配置 `concurrency`，并记录实际观察到的最大并发。所有任务先赋稳定排序键，再按该键收集结果，因此线程调度不会改变分段、算法、边界、失败或摘要顺序。

阶段产物先写入快照目录下的临时阶段目录，再用目录重命名整体替换正式阶段目录。`manifest.json` 记录：

- schema/全局快照版本、运行代次；
- 配置哈希、波形数据指纹、段长、量化指纹；
- 已完成设备、分段游标；
- 特征、打分、阈值、汇总各阶段产物版本；
- 运行状态与当前阶段。

启动时比对快照版本、配置哈希、波形指纹、段长与量化指纹；变化时报告 `E_SNAPSHOT_STALE`，所有阶段目录整体重建，而不是逐文件原地覆盖。指纹仍匹配的缓存条目可继续复用，这保证配置演进与精确失效同时成立。

`SIGINT`/`SIGTERM` 只设置停止请求；当前阶段全部任务完成并落盘后停止后续阶段，报告状态为 `interrupted`，CLI 返回 `0`。

## 注入故障

`config/injections.yaml` 与主管线配置分离，支持：

- `non_finite_sample`：在目标段写入 NaN，特征阶段报告 `E_NONFINITE_SAMPLE`；
- `band_edge_out_of_range`：把目标段最后一个频带边界移到 Nyquist 之外，报告 `E_BAND_INVALID`；
- `peak_factor_window_zero`：把目标算法窗口设为 0，报告 `E_ALGO_PARAM_INVALID`；
- `threshold_equal_boundary`：把有效阈值精确设为目标健康分段的融合分，生成边界诊断。

报告中的 `injection_hits` 逐项包含注入编号、目标、命中状态、实际效果和错误码；失败记录也回填 `injected_id`。单段失败不会中止其他分段或算法。

## 校验与错误码

加载/config 校验支持并继续处理其他合法设备：

`E_CONFIG_INVALID`, `E_SEGMENT_LEN_INVALID`, `E_SEGMENT_MISMATCH`, `E_RATIO_OUT_OF_RANGE`, `E_BAND_INVALID`, `E_THRESHOLD_INVALID`, `E_CONCURRENCY_INVALID`, `E_QUANT_INVALID`。

测试命令与验收命令：

```powershell
py -3.11 -m unittest discover -s tests -v
py -3.11 tools\verify.py
```

实际验证结果见 `docs/RESULTS.md`。最近一次本地执行通过全部 9 项自动化测试和 7 步累积编辑序列验证；四类注入全部命中，注入后共记录 5 个局部失败并继续处理其余分段。

## 增量与全量比较口径

逐段特征摘要、三算法原始分、归一化定点整数、融合总分、异常标记、边界清单、失败清单、注入命中与波形指纹在增量和空白全量之间逐字段比较，并汇总为 `result_fingerprint`。

`reuse/recompute` 是执行路径日志：增量运行有缓存命中，空白全量必然全部为重算，因此这两类日志不可能同时“内容相同”。工具对两类日志分别验证稳定排序，并对所有业务结果逐字段要求完全一致。验证器将该口径明确写入 `artifacts/verification_report.json`。

确定性保证以相同 Python/NumPy 环境和相同配置为前提；实际已验证 Python 3.11 + NumPy 1.26 环境可重复得到相同结果指纹。
