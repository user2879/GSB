# 惯性传感器窗口聚类与密度诊断组件

这是一个从零实现的本地确定性诊断组件，面向多通道陀螺仪波形。它不依赖在线模型、账号、Docker 或外部数据集；波形、配置、编辑流与故障样本全部由固定整数种子生成。

## 能力概览

- 严格 YAML 校验：配置、波形描述、编辑命令流、租约声明；所有非法字段返回 `E_CONFIG_INVALID` 和具体字段路径。
- 逐通道滑窗：固定窗长/窗移；逐窗计算 RMS、超额峭度、频带能量；缺失采样按窗内已见样本均值填补并记录来源。
- 仅当前训练窗口集合的逐列标准化：窗口集合或参数变化会使统计量、观测向量和依赖缓存真实失效。
- 定点整数 KMeans：等距初始质心、固定并列决断、整数距离与质心更新、逐轮量化残差、空簇最远窗口接管证据。
- 高斯核密度基线：配置档位带宽、固定求和顺序、定点密度分值、千分比阈值、边界窗口及高一位宽参考密度。
- 联合租约/快照状态机：真实单调时钟、到期回收、代次推进、孤儿租约接管、四元快照一致性与原子替换。
- 内容指纹增量缓存：逐窗原始特征、标准化统计量、标准化观测、聚类基线、密度基线和快照条目均有稳定缓存键与复用/重算账本。
- 稳定并发输出：`ThreadPoolExecutor` 并发度来自配置；所有任务和结果按窗口序号、阶段名排序。
- 故障隔离：跨窗引用越界、非法核带宽、非有限采样、租约代次回退分别映射业务码；可继续的窗口继续处理。

## 环境

已验证环境：

- Linux（WSL Ubuntu）
- Python 3.11（要求 3.10+）
- NumPy 1.26.4（要求 1.24+）
- PyYAML 6.0.1
- pytest 8.2.0

安装依赖：

```bash
python3 --version
python3 -m pip install "numpy>=1.24" "PyYAML>=6.0" pytest
```

无需安装本包即可运行；命令前加 `PYTHONPATH=src`。在支持现代 PEP 517 的 pip 环境中也可正常构建安装；当前开发机自带 pip 20.3.4，因此验收脚本统一使用 `PYTHONPATH=src`，不依赖 editable 安装。

## 快速运行

```bash
PYTHONPATH=src python3 -m ins_diag validate --config examples/config.yaml
PYTHONPATH=src python3 -m ins_diag run \
  --config examples/config.yaml \
  --state-dir /tmp/ins-diag-state \
  --output /tmp/ins-diag-report.json
```

默认样本包含 2 台设备、6 个通道、43 个最终分析窗、4 个簇。当前样本会输出两个边界窗口：

- `imu-alpha:gy:000096`
- `imu-alpha:gy:000128`

## 验收命令

```bash
# 27 项自动化测试
PYTHONPATH=src python3 -m pytest

# 逐条施加 7 个编辑命令，与干净目录全量重建逐字段对拍
PYTHONPATH=src python3 -m ins_diag verify \
  --config examples/config.yaml \
  --output artifacts/verify-report.json

# 四类独立故障注入
PYTHONPATH=src python3 -m ins_diag faults \
  --root examples/faults \
  --output artifacts/fault-report.json

# 基线 -> SIGKILL -> 同一状态目录重启
bash scripts/crash_experiment.sh

# 一次执行上述全部验收并写汇总
bash scripts/run_acceptance.sh
```

实际结果见 `artifacts/validation-summary.json`、`artifacts/verify-report.json`、`artifacts/fault-report.json` 和 `artifacts/crash-report.json`。

## 样本与配置

- `examples/config.yaml`：窗长/窗移、频带、簇数、定点参数、KDE 带宽与阈值、并发度、状态目录。
- `examples/waveform.yaml`：设备/通道、趋势、周期抖动、平稳噪声、缺失段。
- `examples/edits.yaml`：单窗替换、批次追加、单段删除、簇数调整、核带宽/量化档位/定点位宽切换。
- `examples/lease.yaml`：worker 标识、租约到期毫秒、代次下限。
- `examples/faults/`：四个自包含故障目录，每个目录包含 `expected.json`。

## 关键边界

- 核带宽只接受档位 `"0.10"`、`"0.15"`、`"0.20"`、`"0.30"`、`"0.40"`。
- 定点小数位宽只接受 `8`、`10`、`12`、`14`、`16`；量化档位（整数网格倍数）只接受 `1`、`2`、`4`、`8`。
- 密度分值以当前训练集最大密度的千分比表示；阈值字段也是千分比。边界判定使用定点网格半步。
- 快照必须同时匹配全局版本、配置哈希、数据指纹和租约代次；不一致时输出 `E_SNAPSHOT_STALE`，旧快照不会作为当前状态续跑。
- 增量与全量对拍比较最终逐窗特征、标准化向量、簇归属、密度分值、边界清单、失败清单和规范化的产物顺序。缓存命中/重算状态本身按定义会不同，单独写入缓存账本，不参与最终分析字段对拍。

更详细的设计、业务码与验证记录见 `docs/TECHNICAL.md`。
