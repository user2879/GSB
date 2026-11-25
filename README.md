# ffbudget — 特征新鲜度预算与回填幂等诊断组件

面向推荐冷启动图文排序链路的首版本地组件，覆盖分层特征槽位声明、回填批
幂等裁决、生效时刻视图与陈旧回退、定点整数打分、点数预算调度、内容指纹
精确失效、阶段化快照恢复与诊断报告。

## 环境
- Python 3.11，依赖见 `requirements.txt`（仅 `numpy==1.26.4`）。
- 纯本地文件读写，无容器、无外部服务、无网络访问。

## 快速开始
```bash
python --version
python -m pip install -r requirements.txt
python tools/generate_samples.py
python tools/run_demo.py
python -m unittest discover -s tests -v
python -m ffbudget.cli run --config samples/config.json \
  --batches samples/batches --requests samples/requests.json \
  --state state/run --mode full --out state/result.json
```
Windows PowerShell 使用反引号续行，路径分隔符改为 `\`，详见
`docs/TECHNICAL_DESIGN.md`。

## 模块
- `ffbudget/`：加载校验、权威裁决、陈旧视图、定点打分、预算调度、指纹缓存、快照与报告。
- `samples/`：固定种子 `20261125` 的本地 JSON/NDJSON 样本。
- `tools/`：样本生成与端到端验收演示。
- `tests/`：26 个单元/集成测试。
- `docs/TECHNICAL_DESIGN.md`：业务语义、业务码、定点规则与验证记录。

## 验收要点
`tools/run_demo.py` 自动比对全量重放、增量重放、并发度（1 对 8）、乱序
到达、快照恢复与幂等重投，输出 `state/demo/acceptance_report.json`。
