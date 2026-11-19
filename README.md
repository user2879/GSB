# 语音录音修复工作台（Voice Repair Workbench）

本地、离线、确定性的语音录音修复工作台：固定种子合成 48kHz/16bit 单声道、
24bit 双声道对白/静音/底噪素材与本地有损样本，经“谱减降噪 → 残留噪声门限
→ 增益归一 → 限幅”四阶段管线输出修复 WAV 与逐段指标，并支持工程 schema
v1→v2 迁移、阶段级缓存失效闭包、原子持久化与强制终止恢复。

## 快速开始（Windows）

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m voice_repair_workbench demo --work demo_workspace
```

Linux/macOS 将激活命令换成 `source .venv/bin/activate`。

## 常用命令

- `python -m voice_repair_workbench generate --out demo_workspace/materials`
  仅生成确定性素材与本地 `.lrx` 有损样本。
- `python -m voice_repair_workbench demo --work demo_workspace`
  一键完整验收演示（跨格式/跨版本一致性、迁移幂等、失效闭包、崩溃恢复、错误码）。
- `python -m unittest discover -s tests` 运行 23 个测试。
- `python -m voice_repair_workbench.gui` 启动 Tkinter 界面（无 tkinter 时优雅降级）。

## 文档

详见 [TECHDOC.md](TECHDOC.md)：架构、模块职责、错误码、容差口径、设计取舍与验证结果。
