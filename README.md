# 地形遮挡剔除与流式加载组件（确定性参考实现）

面向大规模地形场景的 Hi-Z 遮挡剔除、视锥可见性、流式驻留调度与误差预算降级
的纯 Python（NumPy）参考实现，配套固定种子程序化地形、相机轨迹、逐帧诊断、
内容哈希与完整测试。设计与数值细节见 `docs/DESIGN.md`。

## 环境

- Python >= 3.10（实测 3.11）
- 固定依赖：`numpy==1.26.4`、`PyOpenGL==3.1.7`、`pytest==8.3.4`
- 离屏 GL：EGL/GLES2 pbuffer 仅用于能力探测，数值路径不依赖 GPU

## 一键运行

```bash
pip install --user -r requirements.txt
python -m terrain_culling.demo --out artifacts/demo --frames 16
python -m pytest tests -q
```

Windows 可运行 `run_demo.bat`，POSIX 可运行 `./run_demo.sh`。

## 产物

输出目录包含：

- `height_<size>.u16.bin` / `material_<size>.u8.bin`：确定性高程与材质
- `manifest_<size>.json`：顺序无关的四叉树块清单（层级、坐标、AABB、高程极差、字节账本）
- `frames_<size>.json`：逐帧完整决策与误差分量
- `summary_<size>.json`：逐帧摘要、降级序列、最终预算、自分歧检查
- `preflight.json` / `run_manifest.json`：环境预检与内容哈希

两次运行的全部产物逐字节一致（含 `run_manifest.json` 的整体 SHA-256）。

## 业务错误码

`E_HEIGHT_RANGE`、`E_BLOCK_MISSING`、`E_HIZ_LEVEL`、`E_DEPTH_BIAS`、
`E_BUDGET_DENIED`、`E_RESIDENT_LEAK`、`E_TRACK_DIVERGE`，定义于
`terrain_culling/errors.py`，异常链路由 `tests/` 全覆盖。
