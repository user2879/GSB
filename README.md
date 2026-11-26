# ChainDiag — 定点嵌入与聚合算子链诊断组件

本地、确定性、无容器、无外部服务的 LLM 推理前算子链诊断实现。组件从固定整数种子生成嵌入表，在定点整数域执行嵌入查表、段池化、多头注意力、层归一化和最终结论固化，并提供 e1 定宽文本/e2 JSON 双版本交换、内容指纹缓存、增量复用、预算调度、异常注入隔离与快照恢复。

## 环境准备

要求 Python 3.10+。当前仓库已在 WSL/Ubuntu 风格环境用 Python 3.11 验证：

```bash
python3 -m venv .venv            # 若 ensurepip 缺失：virtualenv -p python3 .venv
. .venv/bin/activate
pip install -r requirements.txt  # numpy + pytest
```

## 一键验证

```bash
.venv/bin/python -m chaindiag acceptance --output-dir artifacts/verification
.venv/bin/python -m pytest
```

最近一次实际结果：

- `32 passed`（见 `artifacts/verification/pytest.txt`）。
- 验收报告 `artifacts/verification/acceptance_report.json` 中 12 项检查全部为 `true`。
- 覆盖双版本往返、串行/并发一致、8 组随机编辑序列、增量==空白全量、预算待办、算子配额、三类异常注入、高一位宽参考和 SIGKILL 后恢复。

## 常用命令

```bash
# 生成确定性样本、e1/e2 链、旁路、编辑流和注入配置
.venv/bin/python -m chaindiag generate samples

# 校验 e2 JSON
.venv/bin/python -m chaindiag validate --sample-dir samples

# 校验 e1 定宽链并自动读取 samples/chain.e1.sidecar.json
.venv/bin/python -m chaindiag validate \
  --sample-dir samples --chain samples/chain.e1.txt

# 全量运行并写报告
.venv/bin/python -m chaindiag run \
  --sample-dir samples --report artifacts/verification/cli-run.json

# 带异常注入运行
.venv/bin/python -m chaindiag run \
  --sample-dir samples --with-injections \
  --report artifacts/verification/cli-faults.json
```

校验失败统一输出 `E_CHAIN_FORMAT`，JSON 结构错误带 JSONPath，e1 错误带文件行号和字段名。例如：

```json
{
  "code": "E_CHAIN_FORMAT",
  "line": 2,
  "field": "version",
  "location": "samples/invalid_chain.e1.txt:2:version"
}
```

## 目录

- `chaindiag/model.py`：配置、令牌、段边界、算子 DAG、枚举和版本串校验。
- `chaindiag/exchange.py`：e1 15 字符定宽行、e2 JSON、双向转换和降级旁路。
- `chaindiag/fixedpoint.py`：定点舍入、半步决断、量化网格和饱和工具。
- `chaindiag/kernel.py`：嵌入、段池化、多头注意力、整数层归一化。
- `chaindiag/engine.py`：指纹计划、精确失效、逐令牌结论、残差和高一位宽参考。
- `chaindiag/ledger.py`：内容寻址缓存、节点索引和失效原因分类。
- `chaindiag/scheduler.py`：稳定预算准入、算子配额和 `ThreadPoolExecutor`。
- `chaindiag/snapshot.py`：临时目录写入、原子版本切换、配置哈希恢复和硬终止模拟。
- `chaindiag/state.py`：编辑命令流与算子增删后的序号/引用修复。
- `chaindiag/acceptance.py`：一键本地验收和报告生成。
- `tests/`：32 个 pytest 用例，包括 8 组随机编辑序列。
- `samples/`：可重复生成的 JSON/e1 样本。
- `artifacts/verification/`：实际验收报告、完整运行 JSON、快照和测试日志。

## 关键口径

### e1/e2

e1 每行固定 15 字符：

| 字段 | 区间 | 宽度 |
| --- | --- | ---: |
| version | `[0:2]` | 2，固定 `E1` |
| op_seq | `[2:6]` | 4 |
| scale_permille | `[6:12]` | 6 |
| int_bits | `[12:15]` | 3 |

e1→e2 将千分比精确表示为 `permille/1000`，生成 `e1-exact:<permille>/1000` 精度标签，并把 8/16/24/32 位宽映射为 `qint8/qint16/qint24/qint32`。e2→e1 的浮点四舍五入残差、原精度标签、`inputs` 和注意力快照写入 `*.sidecar.json`；带旁路再升 e2 时恢复这些被覆盖字段，验收中的往返结论与原 e2 逐字段一致。

### 定点链

- `frac_bits ∈ {8,16,24}` 是校验层允许的工程档位；高一位宽参考路径使用 `frac_bits + 1`。
- `quant_tier ∈ {q0,q1,q2,q3}` 控制量化网格丢弃位。
- 所有核心累加使用 Python 任意精度整数；舍入返回 `-1/0/+1` 方向和精确半步标记。
- 注意力权重使用整数千分比并由最大余数法保证每个头合计恰为 1000。
- 层归一化按令牌升序执行，维统计按维度升序累加；精确落在量化半步时使用 `half_away_from_zero`，并在令牌结果中设置 `boundary_token=true`。

### 指纹与复用

每个阶段键由以下内容派生：

- 上游阶段键和段成员嵌入键；
- 算子语义指纹；
- 令牌内容指纹；
- `frac_bits` 与 `quant_tier`；
- `exchange_version`。

替换一个令牌时，只有该令牌及其所在段的池化/注意力/归一化/最终结论失效；其他段直接命中。编辑后的当前状态再跑一次时，所有节点均为 `reused`。

### 预算与并发

调度先在单线程中按 `(op_seq, token_seq)` 稳定记账，再把已准入任务交给线程池。点数不足写 `E_POINT_BUDGET_PENDING` 并保留待办；单算子累计点数超过 `operator_quota` 写 `E_OP_QUOTA_EXCEEDED`。串行与任意 `max_workers` 下的结论、缓存键、待办顺序和诊断顺序一致。

### 快照与注入

快照先写 `.staging-*` 临时目录和 `manifest.json`，再 `os.replace` 为 `v00001/...`，最后原子替换 `current` 符号链接。恢复时比较 `config_hash` 与 `exchange_version`；不一致即重建。测试用子进程发送真实 `SIGKILL`，验证未提交暂存目录被清理且上一已提交快照仍可恢复。

注入项 `INJ-DIM-001`、`INJ-NAN-002`、`INJ-SEG-003` 分别覆盖维度不符、非有限特征和段边界越界。报告 `artifacts/verification/faults.json` 证明三个编号均命中指定令牌，其余令牌的向量与缓存键保持不变。
