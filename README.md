# 本地合成工单“升级仲裁”领域数据编辑与操作应用

纯 Python（3.10+，仅诊断直方图依赖 numpy）实现，无容器、无外部服务。

## 快速开始

```bash
pip install numpy
python3 -c "from escalation_arbiter.samplegen import generate; generate('samples')"
python3 -m escalation_arbiter --samples samples --workdir workdir
python3 -m pytest tests/ -q
```

运行后产物均位于工作区内部：

- `workdir/cache/`：四阶段内容寻址缓存
- `workdir/tmp/`：快照提交临时目录
- `workdir/snapshot/`：原子替换的阶段快照与 manifest
- `workdir/exchange/`：v1/v2/v3 三种交换文件与旁路 sidecar
- `workdir/diagnostic_report.json`：注入命中、失败清单、偏差直方图与还原统计

## 样本输入（`samples/`，确定性生成）

- `evidence.jsonl`：逐行工单证据
- `rules.json`：类型权重、角色可信度、互斥/支持关系、分档阈值、原因码字典
- `config.json`：定点位宽/量化档位、半衰期、扣减/加成强度、融合方式、零下界、
  交互预算、并发数、快照间隔、可信度声明区间
- `commands.json`：15 条覆盖全部编辑类型与三版本切换的交互命令
- `injections.json`：四类真实输入故障注入清单

## 模块导引

- 定点与半衰期：`escalation_arbiter/fixedpoint.py`
- 三阶段计算：`escalation_arbiter/stages.py`
- 三版本交换格式：`escalation_arbiter/versions/`
- 增量失效与缓存：`escalation_arbiter/cache.py`、`references.py`、`engine.py`
- 设计细节：`docs/DESIGN.md`
- 实测结果：`docs/VERIFICATION.md`
