# 确定性批次归属审计台

面向 Windows 本地环境的推理批处理调度审计台。它处理乱序到达、分批和同槽重复投递的样本，生成可由第三方独立复算的批次归属编号，并把每个已提交槽位持久化到 SQLite 执行账本。

项目不依赖 Docker、外部数据下载或在线服务。样本、MLP 参数和投递序列均由本地整数随机种子确定性构造。

## 核心能力

- 固定随机种子生成逐位一致的输入矩阵与两层前馈网络参数。
- 强校验 JSON 投递序列；出错时以 `E_FEED` 整体拒绝，并给出首个错误字段路径与到达序号。
- 按 `arrival_seq` 规范排序，按 `batch_slot` 分组；同槽重投保留最小到达序号并登记覆盖记录。
- 归属编号只依赖批次槽位、组内名次、样本标识、模型参数摘要与编号口径版本。
- 每个非空槽位执行 NumPy 前向推理，记录批次输出摘要并与每条归属编号绑定。
- 每个槽位的编号集、输出摘要、覆盖记录和进度游标在同一 SQLite 事务中提交。
- 进程强制终止后只承认已提交槽位；未提交槽位标记为未知并要求重做，终止时不会生成退出报告。
- 投递序列、模型参数或编号口径变化时版本号单调递增，当前报告整体切换到新版本，旧编号不参与。
- 独立审计工具只读账本与原始投递序列，自行重放排序、分组、编号和推理，按槽位逐项比较。
- Tkinter 本地工作台与 CLI 共用同一核心逻辑。

## 环境准备（Windows）

1. 安装 Python 3.10 或更高版本（python.org 安装包默认包含 Tkinter）。
2. 在项目根目录执行：

```bat
py -3.10 -m pip install -r requirements.txt
py -3.10 -c "import numpy, sqlite3, tkinter; print(numpy.__version__, sqlite3.sqlite_version, tkinter.TkVersion)"
```

若 `tkinter` 不可用，请在 Python 安装器中启用 “tcl/tk and IDLE” 组件。

## 快速开始

生成确定性的投递序列、样本矩阵与模型参数：

```bat
py -3.10 -m batch_audit.cli init --workspace workspace --seed 20261120 --samples 12 --input-dim 6 --hidden-dim 8 --slots 0,1,2,3,4
```

执行调度并生成 `workspace/report.json`：

```bat
py -3.10 -m batch_audit.cli run --workspace workspace
```

独立复算审计：

```bat
py -3.10 -m batch_audit.cli audit --workspace workspace
```

启动本地 GUI：

```bat
py -3.10 -m batch_audit.cli gui --workspace workspace
```

也可双击 `setup.bat`、`init.bat`、`run.bat`、`audit.bat`、`gui.bat`。CLI 输出为结构化 JSON；成功审计返回 `AUDIT_OK`，部分可审计返回 `AUDIT_PARTIAL`，不一致返回 `E_AUDIT` 与退出码 3。

## 演示工作区

`init` 生成以下本地文件：

| 文件 | 用途 |
| --- | --- |
| `feed.json` | 乱序投递序列；含同槽重投、准入失败和空槽位 |
| `dataset.npz` | 确定性样本标识、输入矩阵与生成配置 |
| `model.npz` | 两层 MLP 的权重与偏置 |
| `manifest.json` | 工作区参数与文件索引 |
| `ledger.sqlite3` | 执行账本，首次 `run` 创建 |
| `snapshot.json` | 当前版本与槽位进度快照 |
| `report.json` | 仅在完整执行后生成 |
| `structured.jsonl` | 追加式结构化运行日志 |

默认 12 个样本、5 个槽位。演示投递包含两个同槽重投、两个准入失败条目，槽位 `4` 为空并登记 `E_EMPTY`。

## 投递格式

```json
{
  "schema": "batch-feed-v1",
  "entries": [
    {
      "sample_id": "S0000",
      "arrival_seq": 1,
      "batch_slot": 0,
      "admitted": true
    },
    {
      "sample_id": "S0010",
      "arrival_seq": 13,
      "batch_slot": 0,
      "admitted": false,
      "reject_reason": "policy_denied"
    }
  ]
}
```

校验规则：

- `sample_id`：非空字符串；成功条目必须存在于本地确定性样本集，准入失败条目可使用外部标识。
- `arrival_seq`：整数且全局唯一；JSON 布尔值不算整数。
- `batch_slot`：受支持槽位集合内的整数。
- `admitted`：严格布尔值。
- `reject_reason`（可选，也接受 `reason`）：准入失败原因；缺失时登记为 `admission_rejected`。
- 成功样本允许在同一槽位重复投递：保留 `arrival_seq` 最小者，其余登记为覆盖条目。成功样本若跨槽位重复，会形成归属歧义，以 `E_FEED` 整体拒绝。

这一解释把“成功样本唯一”落实为“去重后的成功归属唯一”，同时满足同批重投必须可处理的业务要求。

## 报告与编号

归属编号示例：

```text
BA-V1:S0:R000001:fe81b1eb39cf0e4e8812
```

组成规则：

```text
<numbering_version>:S<batch_slot>:R<rank:06d>:<sha256_prefix_20>
```

完整哈希输入与摘要口径见 `docs/DESIGN.md`。编号不依赖文件书写顺序、Python 字典遍历顺序、槽位处理顺序或批次成员排列。

`report.json` 包含：

- 当前版本、输入摘要、参数摘要、配置摘要与编号口径。
- 每个槽位的状态、成员与名次、归属编号、样本输出摘要、批次输出摘要、覆盖和失败条目。
- `E_EMPTY` 空槽位、按槽位和原因分列的准入失败清单。
- 无法归属样本清单及 `ADMISSION_FAILED:*` / `NOT_DELIVERED` 原因。
- 版本替换原因、受影响样本数和槽位。
- 中断恢复信息：被中断的尝试、重做前未知槽位、本次重做槽位。

## 强制终止与恢复

验证时可注入强制退出（绕开所有清理和退出报告）：

```bat
py -3.10 -m batch_audit.cli init --workspace crash-demo
py -3.10 -m batch_audit.cli run --workspace crash-demo --crash-after-commits 2
echo %ERRORLEVEL%
```

预期退出码为 `99`，且工作区没有 `report.json`。此时：

- 已提交槽位保留并可审计。
- 其余槽位显示 `pending`；重启时标记为 `unknown` 并重做。
- `snapshot.json` 仍为未完成状态，不声称存在终止退出报告。

重启并复核：

```bat
py -3.10 -m batch_audit.cli status --workspace crash-demo
py -3.10 -m batch_audit.cli audit --workspace crash-demo
py -3.10 -m batch_audit.cli run --workspace crash-demo
py -3.10 -m batch_audit.cli audit --workspace crash-demo
```

中断后的首次审计返回 `AUDIT_PARTIAL`，明确列出不可审计槽位；续跑完成后返回 `AUDIT_OK`。

## 版本演进

- 版本指纹由投递摘要、输入摘要、参数摘要、编号口径和配置摘要组成。
- 任一指纹变化，或快照版本与账本当前版本不一致，都会创建新版本，旧版本标为 `superseded`。
- 报告和审计只读取最新版本；旧编号不会增量合并到当前结果。
- 新版本号取“账本最大版本号、快照版本号”的较大值加一，保证单调递增。
- 新版本报告列出 `affected_samples` 与 `affected_slots`。

模拟编号口径演进：

```bat
py -3.10 -m batch_audit.cli run --workspace workspace --numbering-version BA-V2
```

## 测试

```bat
py -3.10 -m unittest discover -v
```

测试覆盖确定性资产、乱序/重投分组、槽位处理顺序不变性、编号绑定、SQLite 强制终止恢复、版本整体替换、独立审计与篡改定位。本次实测结果见 `docs/VALIDATION.md`。

## 退出码

| 退出码 | 含义 |
| --- | --- |
| `0` | 成功或审计通过 |
| `2` | `E_FEED`，投递序列整体拒绝 |
| `3` | `E_AUDIT`，首个不一致 |
| `4` | `E_STATE`，账本、快照、文件或配置错误 |
| `99` | 测试用强制终止注入 |

详细规范见 `docs/DESIGN.md`。
