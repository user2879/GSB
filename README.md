# renderdiag

本地确定性、多模态条件渲染诊断组件。提示词、PNG 图像和分块运动矢量表经定点编码与融合后，驱动初始化、逐去噪步、逐帧解码和按分块行 PNG 输出；所有阶段使用内容指纹缓存键，可在编辑后精确复用未受影响产物，并在交互预算内严格区分陈旧摘要与完整最新结果。

工程初始不依赖预置样本：`sample` 命令按固定整数种子生成 PNG、运动表、提示词、配置、编辑命令流和注入清单。运行不使用网络服务、GPU、外部数据集或容器。

## 快速开始

```bash
python3.11 -m virtualenv .venv   # 本机缺少 ensurepip 时使用 virtualenv
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/acceptance.py --out artifacts/acceptance
```

本机实际验证：Python `3.11.0rc1`、NumPy `1.26.4`、Pillow `11.3.0`、PyYAML `6.0.4`，20 项单元测试通过，完整验收通过。结果见 `artifacts/acceptance/report.json`、`artifacts/acceptance/SUMMARY.md` 和 `artifacts/acceptance/unittest.log`。

## 常用命令

```bash
# 生成确定性本地样本
.venv/bin/python -m renderdiag sample --out artifacts/sample

# 校验 YAML、受控词表、PNG 容器/分辨率/像素模式和命令流
.venv/bin/python -m renderdiag validate \
  --config artifacts/sample/config.yaml \
  --commands artifacts/sample/commands.jsonl

# 渲染并逐条执行编辑命令，输出帧与 JSON 报告
.venv/bin/python -m renderdiag render \
  --config artifacts/sample/config.yaml \
  --workspace artifacts/cli-run \
  --commands artifacts/sample/commands.jsonl \
  --report artifacts/cli-run/report.json
```

## 模块

- `renderdiag/config.py`：配置 YAML、枚举、维度和分块约束校验。
- `renderdiag/resources.py`：受控词表、PNG 头与像素、运动表、命令流、注入清单校验。
- `renderdiag/codec.py`：文本词袋、图像统计、运动编码和定点条件融合。
- `renderdiag/pipeline.py`：潜变量初始化、逐去噪步、投影、分块行与帧解码。
- `renderdiag/store.py`：内容寻址产物目录和整体版本化快照。
- `renderdiag/engine.py`：失效判定、稳定并发调度、点预算、待办、摘要与会话恢复。
- `tests/`：20 项确定性、增量、调度、恢复与故障注入测试。
- `scripts/acceptance.py`：13 条编辑序列的端到端验收和不变量报告。

设计与语义细节见 `docs/DESIGN.md`，本次实测结果见 `docs/RESULTS.md`。
