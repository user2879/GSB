# Windows 运行指南

本项目只依赖 `numpy` 与 `PyYAML`，不使用 Docker、外部服务或网络样本。所有样本均由固定种子 `239` 确定性生成。

## 1. 环境

在 PowerShell 或 CMD 中进入交付目录：

```powershell
py -3.11 --version
py -3.11 -m pip install numpy PyYAML
```

也可使用 Python 3.12；若 `py` 启动器不可用，将下面命令中的 `py -3.11` 替换为 `python`。
若本机 3.12 由 uv 管理且基础环境提示 `externally-managed-environment`，先执行 `py -3.12 -m venv .venv` 或 `uv venv --python 3.12`，再激活 `.venv` 后安装依赖。本次交付已在真实 Windows 11 + CPython 3.12.16 的独立 venv 中跑通。

## 2. 一键生成样本、构建、对拍并测试

```powershell
py -3.11 run_validation.py --root sample_workspace
```

成功时最后输出包含：

```json
"overall_success": true
```

并生成：

- `docs/VALIDATION_REPORT.md`：实际测试与验证结果。
- `sample_workspace/outputs/validation_summary.json`：机器可读汇总。
- `sample_workspace/outputs/equivalence_report.json`：14 条命令的增量/全量逐字段对拍。
- `sample_workspace/outputs/invalidation_report.json`：每条命令的失效、复用、缓存键和判定依据。
- `sample_workspace/outputs/fault_injection_report.json`：9 个故障注入项及命中证据。
- `sample_workspace/outputs/failure_manifest.json`：非法语料、配置和回执的失败清单。
- `sample_workspace/outputs/unified_ledger.jsonl`：三版本归一后的统一账本。

## 3. 分步命令

```powershell
py -3.11 -m dense_index.cli generate-sample --root sample_workspace
py -3.11 -m dense_index.cli validate-inputs --root sample_workspace
py -3.11 -m dense_index.cli build --root sample_workspace
py -3.11 -m dense_index.cli replay --root sample_workspace
py -3.11 -m dense_index.cli failure-manifest --root sample_workspace
py -3.11 -m dense_index.cli faults --root sample_workspace
py -3.11 -m unittest discover -s tests -v
```

## 4. 约定目录

```text
sample_workspace/
  config/
    build_config.yaml
    vocabulary.json
    injections/
  data/
    paragraphs.jsonl
    queries.json
  commands/
    command_stream.json
  receipts/
    receipts_v1.json
    receipts_v2.json
    receipts_v3.json
  fixtures/
    corpus_invalid.jsonl
    receipt_invalid_v1.json
  outputs/
    incremental/snapshots/
    injections/
```

## 5. 关键业务码

- `E_CORPUS_INVALID`：JSONL、段落 ID、文档 ID、序号或受控词表校验失败。
- `E_CONFIG_INVALID`：窗口、重叠、维度、桶位、量化枚举或阈值非法。
- `E_RECEIPT_INVALID`：回执版本未知、字段缺失、类型错误或 JSON 截断。
- `E_VECTOR_NONFINITE`：投影/定点向量出现 NaN 或无穷分量。
- `E_BOUNDARY_FLIP`：高低位宽或决胜规则导致边界判定翻转；`cause` 为 `quantization_truncation` 或 `tie_rule`。
- `E_POINT_BUDGET_PENDING`：本批预算耗尽，仍有稳定排序后的待办块。
- `E_SNAPSHOT_RECOVERY`：快照截断后从最近有效整体提交恢复。

## 6. 确定性与 Windows 兼容

- 哈希使用稳定 FNV-1a/SHA-256，不使用进程随机化的 Python 内置 `hash()`。
- 投影矩阵由配置指纹派生的 PCG64 种子生成。
- 所有 JSON/JSONL 均以 UTF-8、LF 换行写出；路径全部经过 `pathlib.Path`。
- 快照先写入独立 `.staging-*` 目录，再整体重命名为 `snapshot-vXXXXXX`，最后用 `os.replace` 原子更新 `CURRENT` 指针。
