# 推理服务身份与令牌盖章组件（本地单机）

面向本地 Windows / 跨平台环境的推理后端身份摘要与推理令牌盖章、
验章、账本与重放组件。纯本地确定性数据，不下载模型、不依赖外部服务、
不使用容器。

## 快速开始

```bash
python -m pip install -r requirements.txt
python scripts/generate_samples.py
python scripts/demo.py
python -m unittest tests.test_all -v
python scripts/audit.py --spec samples/spec.base.json --ledger run/main.db
```

Windows PowerShell 下把 `/` 换成 `\`，或直接使用正斜杠（Python 支持）。

## 异常注入

```bash
python scripts/demo.py --inject corrupt-spec       # E_SPEC
python scripts/demo.py --inject tamper-identity    # E_IDENTITY
python scripts/demo.py --inject bad-snapshot       # E_SNAPSHOT + 整体替换
```

## 样本文件

| 文件 | 含义 |
| --- | --- |
| `samples/spec.base.json` | 基线 3 后端 / 8 请求 |
| `samples/spec.evolved-identity.json` | alpha 身份版本 1→2 |
| `samples/spec.evolved-token-spec.json` | alpha 令牌口径 1→2 |
| `samples/spec.evolved-quant.json` | beta 量化 fp32→fp16 |
| `samples/spec.corrupt.json` | gamma hidden_dim=0 损坏声明 |

## 文档

- 设计与规则：`docs/design.md`
- 实际验证记录：`docs/verification.md`
- 任务书原文：`README.md`
