# 依存解析结果双版本编辑应用（depedit）

面向中文依存句法分析结果的领域数据编辑与操作应用，解决 v1 定宽文本与 v2 JSON
两种工程交换版本之间的互操作、版本兼容与可追溯降级问题。

## 快速开始

环境：Python 3.11 + NumPy 2.2.6（依赖见 `requirements.txt`），无需容器。

```bash
python3 -m depedit.cli --data-dir data/sample --out-dir out generate
python3 -m depedit.cli --data-dir data/sample --out-dir out fingerprints
python3 -m depedit.cli --data-dir data/sample --out-dir out parse --source v1
python3 -m depedit.cli --data-dir data/sample --out-dir out --injections config/injections.json parse --source v1
python3 -m depedit.cli --data-dir data/sample --out-dir out convert --direction up
python3 -m depedit.cli --data-dir data/sample --out-dir out edit
python3 -m depedit.cli --data-dir data/sample --out-dir out replay
python3 -m pytest
python3 scripts/run_verification.py
```

## 能力概览

- 双版本解析器与统一中间表示（1 起始序号，头词 0 为根，命名关系，0..1 浮点得分）。
- 四类编辑：改头词、换关系、句内头词整体平移、标记重解析；结构非法抛 `E_HEAD_STRUCTURE` 且不写入。
- v1/v2 双向转换，缺失头词旁路元数据与三类降级损失（残差、未映射关系原名、编号折算）。
- 词序列 / 模型 / 映射 / 版本号派生的分层指纹与精确到弧的增量失效，未失效句子复用产物。
- 临时目录 + 原子整体替换的版本化快照，恢复不一致抛 `E_SNAPSHOT_STALE` 并保留旧快照。
- 六类故障注入隔离、线程池并发稳定排序、跨版本往返诊断与编辑序列确定性回放。

详细架构、数据契约与验证结果见 `docs/TECHNICAL.md`。
