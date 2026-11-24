# 本地确定性旋转机械振动异常检测管线

面向本地确定性合成旋转机械振动信号的无监督异常检测数据处理管线与结果验证工具。
核心是跨分段、跨窗双重耦合的**特征谱构造与精确失效传导**，并支持定点量化、
三算法融合、特征点预算、并发调度与阶段快照续算。

## 环境准备

```bash
python --version            # 需要 Python 3.11 或 3.12
pip install -r requirements.txt
python -c "import numpy, yaml"
```

## 快速开始

```bash
# 1. 生成本地确定性 YAML 配置与振动描述（无需联网）
python -m vibdet init --workspace ws

# 2. 全量运行；任意配置项可用点路径覆盖
python -m vibdet run --workspace ws \
  --override window.length=200 --override window.hop=100 \
  --override spectrum.order=3 --override quantization.frac_bits=10

# 3. 编辑后增量运行
python -m vibdet edit --workspace ws --kind replace-window \
  --device DEV-A --index 2 --value 3.0
python -m vibdet edit --workspace ws --kind delete-segment \
  --device DEV-A --segment-id A-seg2
python -m vibdet run --workspace ws

# 4. 端到端验收与单元测试
python tools/acceptance.py --workspace artifacts/acceptance_ws
python -m pytest tests/ -q
```

## 产物

- `output/results.json`：设备、窗、特征谱位置、三算法得分、定点判定、分段回映、
  失败清单与缓存统计。
- `output/diagnostics.json`：状态、耗时、退出信号、待处理队列与快照诊断。
- `snapshots/manifest.json` 与 `snapshots/<stage>/v<N>/`：版本化阶段快照。
- `cache/`：窗/谱/算法/判定四层内容寻址缓存。

## 深入阅读

- 架构、公式、缓存键、失效传导、并发与快照机制见 `docs/TECHNICAL_DESIGN.md`。
- 错误码：`E_DEVICE_UNKNOWN`、`E_WINDOW_INVALID`、`E_SPECTRUM_TOO_SHORT`、
  `E_SAMPLE_NONFINITE`、`E_POINT_BUDGET_PENDING`。
