# 本地确定性文生图 / 图生图多阶段渲染工作台

纯本地、确定性、无 GPU 与网络依赖的渲染管线与结果验证工具。

- 入口：`python run_workbench.py {render|full|baseline|inspect} ...`
- 样本生成：`python tools/generate_samples.py --out samples`
- 单元/集成测试：`python -m unittest discover -s tests`（30 个）
- 端到端验证：`python tools/validate_workbench.py`（79 项断言）
- 技术设计：`docs/TECHNICAL_NOTES.md`
- 使用说明：`docs/USAGE.md`
- 依赖：`requirements.txt`（numpy、PyYAML）

核心能力：受限词表哈希词袋编码、指纹派生的确定性潜变量与去噪、图生图早期步
替代、线性投影 PNG 解码、基于内容指纹的精确失效传播、局部补丁容差 no-op、
原子快照与强制终止恢复、线程池并发与串行逐像素一致。
