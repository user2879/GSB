# 本地离屏多介质透射渲染引擎

纯 Python + NumPy 的确定性离屏渲染器：把含透明介质（雾、玻璃板、有色水体、
高密度吸收板）与点光源的场景渲染为二维灰度 PGM 图像，并对单精度数值误差、
步长与截断阈值的稳定性、分块中断恢复给出可复核评估。无需网络、GPU 或外部资源。

## 环境要求

- Python 3.10+
- NumPy 1.x 或 2.x（开发环境验证：Python 3.11 + NumPy 2.4.6，Windows/Linux 均可）

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
# 1. 生成确定性本地样本（scene.json / medium_grid.npz / expected_transmittance.json）
python -m offscreen_render.cli generate \
    --sample-dir samples --state-dir state --output-dir output

# 2. 渲染（含双/单精度误差、截断统计、步长/阈值稳定性评估）
python -m offscreen_render.cli render --scene samples/scene.json
```

产物：

- `output/grayscale.pgm`：P5 二进制灰度图（0..255，原子写出）
- `output/report.json`：截断统计、首次超差点定位、稳定性趋势结论
- `state/tile_*.json`：分块级持久化状态（含版本标识与内容摘要）
- `state/archive/`：版本变更或损坏块的归档

## 一键本地验收

```bash
python verify_local.py
```

覆盖确定性样本、解析期望透射率、中断退出码 130/E_4001、缺失块恢复、
分块摘要比对、版本变更归档、非法输入 E_xxx 零副作用，共 28 项检查。
可通过 `--width/--height/--supersample` 不改源码更换输入规模。

## 单元测试

```bash
python -m unittest discover -s offscreen_render/tests
```

38 个测试覆盖几何归属、透射率解析对照、双/单精度、校验、持久化、调度与样本确定性。

## 进一步阅读

- `TECHNICAL_NOTES.md`：架构、算法、误差传播机理、错误码表与复现实验。
- 场景 JSON 字段与取值约束见 `offscreen_render/scene.py`。

## 退出码约定

| 码 | 含义 |
| --- | --- |
| 0 | 成功 |
| 1 | 业务错误（JSON 内含 `E_xxx` 业务码） |
| 130 | 分块边界收到中断（`E_4001`），未完成块不留下可误认为完成的结果 |
