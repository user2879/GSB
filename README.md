# 多波段瓦片异常检测

这是一个仅依赖 NumPy、Pillow 和 Python 标准库的确定性多波段瓦片异常检测实现。程序在 Linux 上生成 3×3 本地瓦片样本，对跨瓦片窗口计算光谱指数、纹理梯度与高程坡度，经定点量化、逐列稳健标准化、马氏距离和 KDE 得到异常分，再融合相邻瓦片边界并输出像素级异常 PNG。

## 环境

- Python ≥ 3.10，当前验证环境为 Python 3.11.0rc1
- NumPy 1.26.4
- Pillow 10.3.0
- Linux（WSL Ubuntu），16 个逻辑 CPU

创建隔离环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

如果系统缺少 `ensurepip`，可使用本机已安装的 `virtualenv`：

```bash
python3 -m virtualenv --system-site-packages .venv
```

## 快速开始

```bash
./scripts/generate_sample.sh
./scripts/run_demo.sh
```

等价命令：

```bash
.venv/bin/python -m tile_anomaly generate --output sample --seed 20261126
.venv/bin/python -m tile_anomaly run --config sample/config/detection.yaml --reset --json
```

输出位置：

- `sample/input/tiles/<tile_id>/{red,green,blue,elevation}.png`：分层波段
- `sample/input/manifest.json`：行列号、邻接关系、波段集合版本
- `sample/config/detection.yaml`：窗口、量化、检测、融合与调度配置
- `sample/config/fault_injection.json`：四类真实输入故障注入
- `sample/input/edit_commands.json`：确定性编辑命令流
- `sample/artifacts/output/pixel_anomaly.png`：最终异常栅格
- `sample/artifacts/state/current/`：原子提交的最新阶段快照

`detection.yaml` 使用 JSON 语法书写，它是 YAML 的合法子集；加载器也支持环境中安装 PyYAML 时解析一般 YAML。

## 应用编辑与增量运行

```bash
.venv/bin/python -m tile_anomaly apply-edits \
  --config sample/config/detection.yaml \
  --edits sample/input/edit_commands.json
.venv/bin/python -m tile_anomaly run --config sample/config/detection.yaml --json
```

运行器会从 `state/current` 恢复缓存、待办窗口和预算。相同内容不重复计算；阈值变化复用窗口特征、稳健统计量和异常分；波段替换只重算引用该瓦片的窗口特征。

## 故障隔离

```bash
.venv/bin/python -m tile_anomaly run \
  --config sample/config/detection.yaml \
  --fault-config sample/config/fault_injection.json --reset --json
```

四类故障分别记录在瓦片级事件中，业务码均为 `E_TILE_FORMAT`，其余可用瓦片继续完成检测。窗口上下文涉及故障瓦片时记为 `E_WINDOW_CONTEXT_UNAVAILABLE`，对应像素保持未知值 128。

## 测试

```bash
.venv/bin/python -m unittest discover -v
./scripts/run_verification.sh
```

验证脚本会生成：

- `artifacts/verification/tests.log`
- `artifacts/verification/baseline_run.json`
- `artifacts/verification/fault_run.json`
- `artifacts/verification/summary.json`
- `artifacts/verification/fault-sample/`：独立故障场景，避免覆盖基线快照

## 关键设计

- 窗口核心区互不重叠，每个核心区只属于一个中心瓦片；完整上下文允许跨瓦片。
- 四维观测量以显式定点公式量化，逐维记录步长、整数结果与残差。
- 稳健中心使用整数双中位数，尺度使用整数 MAD 与固定 1.4826 系数。
- 马氏距离采用逐列稳健标准化后的对角 covariance；KDE 带宽来自配置。
- 分数缓存键不包含阈值，因此阈值编辑不会使特征、统计量或异常分失效。
- 融合结论引用两侧瓦片摘要缓存键、两侧窗口分数键和重叠几何。
- 所有快照先写临时目录，再整体改名为版本目录，最后原子替换 `current` 符号链接。

详细算法、缓存依赖和调度规则见 [DESIGN.md](DESIGN.md)，实际运行结果见 [VERIFICATION.md](VERIFICATION.md)。
