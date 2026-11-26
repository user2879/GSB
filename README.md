# 定点整数多模态条件渲染管线

当文本、图像与运动三类条件在同一帧争夺有限去噪步数预算时，本系统在纯定点
整数域内完成条件编码、融合、潜变量初始化、逐去噪步生成、逐帧解码与按分块行
输出，并通过内容指纹账本实现精确增量失效、阶段化快照与崩溃恢复，保证结果
可解释、可复核、可重复。

## 快速开始

```bash
python3 scripts/check_env.py
python3 -m fixedpoint_render.cli init ./workspace --frames 5
python3 -m fixedpoint_render.cli validate ./workspace
python3 -m fixedpoint_render.cli render ./workspace
python3 -m fixedpoint_render.cli report ./workspace
python3 -m unittest discover -s tests
python3 scripts/acceptance.py
```

## 文档

- `docs/ARCHITECTURE.md`：架构、定点运算、预算仲裁、帧间依赖、账本失效、
  快照恢复与错误码全集。
- `docs/USAGE.md`：环境准备、命令、测试与实际验证记录。

## 代码布局

核心实现位于 `fixedpoint_render/`，单元测试位于 `tests/`，验收与环境检查
脚本位于 `scripts/`，验收记录位于 `artifacts/`。无 Docker、无外部网络服务、
无 GPU 依赖；样本由固定整数种子在运行时确定性生成。
