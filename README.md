# 点云与三维重建编辑应用（pcrecon）

面向点云与三维重建显示的领域数据编辑应用：固定种子程序化生成 12 万带法线与
置信度的点（立方体房间 + 地面），离屏渲染点云与三角网格，支持点剔除、区域框选、
法线重估、重建参数调整，增量重算受影响曲面片并与全量重建做一致性核对，
产出可复算的增量重建报告、逐帧耗时报告与 BLAKE2b 内容哈希。

详细设计见 `docs/DESIGN.md`。

## 环境准备（Windows / Python 3.10+）

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

依赖版本固定在 `requirements.txt`：numpy 2.2.6、PyQt5 5.15.11、
PyOpenGL 3.1.9、PyOpenGL-accelerate 3.1.9。

## 首次 OpenGL 核对

```powershell
$env:QT_QPA_PLATFORM="offscreen"; $env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts\check_gl_context.py
```

成功打印 renderer/GL 版本/FBO id；失败输出 `E_GL_CONTEXT` 结构化错误（退出码 2）。

## 诊断入口

```bash
python -m pcrecon generate      # 生成 data/ 下确定性样本（首次运行产出）
python -m pcrecon check-gl      # 离屏 GL/FBO 核对
python -m pcrecon run --undo    # 固定编辑序列 + 一致性核对 + 撤销 + 报告
python -m pcrecon crash-demo    # 强制终止恢复 / 孤儿回收 / 不伪造退出报告
python -m pcrecon hash-verify   # 二次运行比对报告逐字节哈希
```

一键验收：Windows 用 `scripts/acceptance.ps1`，Linux/WSL 用 `scripts/acceptance.sh`。

## 产出物

- `data/`：`points.pcrc`、`normals.pcrc`、`manifest.json`、相机轨迹、编辑定义
- `reports/incremental_report.json`：逐编辑失效闭包、重算/复用、一致性偏差、哈希
- `reports/frame_report.json`：逐帧耗时、降级原因、密度/延迟/跳帧记录
- `ledger/`：原子提交日志、版本清单、提交标记、正常退出报告
