# graymirror — 灰度路由与流量镜像中间件（定点误差传播）

面向本地 Windows/跨平台环境的无容器、无网络中间件：以整数万分比声明灰度权重与
镜像采样率，按会话缩放因子量化为定点整数，基于客户端键/请求标识的稳定哈希完成
灰度分流与镜像采样，并在报告中给出可复现的量化误差界、逐级误差传播、贡献排序、
版本整体替换、快照恢复与重放校验。

## 环境准备

- Python 3.11+；运行时仅用标准库，样本生成依赖 NumPy。
- Windows 使用 python.org 官方安装包即自带 Tkinter（可选查看器）。
- 依赖安装：`python -m pip install -r requirements.txt`。

## 快速开始

```
python -m graymirror.samplegen samples
python -m graymirror.cli run --rules samples/rules.json \
  --stream samples/stream.ndjson --scale-exponent 2 \
  --snapshot samples/snapshots/session-k2.json --report samples/report-k2.json
python -m graymirror.cli restore --rules samples/rules.json \
  --stream samples/stream.ndjson --snapshot samples/snapshots/session-k2.json
python -m graymirror.cli inject --kind weight_out_of_range
python -m graymirror.cli inject --kind timestamp_regression
python -m graymirror.cli inject --kind snapshot_scale_missing
python -m unittest discover -s tests
```

一键演示：Windows `scripts/run_demo.ps1`，Linux/macOS `scripts/run_demo.sh`。

## 业务码

| 码 | 含义 | 触发入口 |
|---|---|---|
| `E_RULE` | 规则文件校验失败 | 权重越界等注入/加载 |
| `E_STREAM` | NDJSON 请求流校验失败 | 时间戳回退等注入/加载 |
| `E_GRID` | 定点网格/缩放因子失败 | 指数越界、快照缺缩放因子 |
| `E_VERSION` | 快照与当前会话版本/摘要不一致 | restore |
| `E_REPLAY` | 恢复重放与快照逐项不一致 | restore + 篡改快照 |

完整定点数学推导、误差传播与实测数据见 `docs/DESIGN.md`。
