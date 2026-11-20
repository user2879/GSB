# 多环境部署一致性验证工作台

这是一个只依赖 Python 标准库的本地工作台，面向 Windows 上的多环境部署配置校验。它从本地 JSON 读取环境、配置源和数值消费任务，构建逐环境逐键的来源链，执行整数定点聚合，拆分误差界与归因，并把结论、版本、快照和恢复状态持久化到工作区。

## 运行环境

- Python 3.11 或更高版本。
- `tkinter` 与 `sqlite3` 标准库。Windows 官方 Python 安装包默认包含二者。
- 不需要网络访问，不依赖第三方包或容器。

当前交付验证环境是 Linux/Python 3.11.0rc1，其中的 Python 未编译 `tkinter`。因此核心、CLI、SQLite、快照和全部自动化测试已实测；Tk 界面已做语法与静态实现检查，但需在带 Tk 的 Windows 官方 Python 上运行。

## 快速开始

Windows 命令提示符：

```bat
run_workbench.cmd gui --spec data\sample_spec.json --state-root .workbench
run_workbench.cmd run --spec data\sample_spec.json --state-root .workbench
run_workbench.cmd replay --spec data\sample_spec.json --state-root .workbench
run_workbench.cmd recover --spec data\sample_spec.json --state-root .workbench
```

等价的 `py` 命令：

```bat
py -3 -m deploy_workbench gui --spec data/sample_spec.json --state-root .workbench
py -3 -m deploy_workbench run --spec data/sample_spec.json --state-root .workbench
```

不指定命令时默认打开 Tk 界面。所有 SQLite、日志、注入输入和快照都在 `--state-root` 下创建：

```text
.workbench/
  ledger.sqlite3
  logs/workbench.jsonl
  snapshots/latest.json
  snapshots/snapshot-run-<id>.json
  injected/
```

## 声明文件

完整可运行样本位于 `data/sample_spec.json`。主要结构如下：

```json
{
  "format_version": 1,
  "scale_power": 3,
  "overflow_policy": "saturate_int64",
  "error_dominance_threshold": 0.5,
  "source_type_bias": {
    "baseline": 2,
    "file_override": 1,
    "runtime_injection": 0
  },
  "sources": [
    {"id": "base", "type": "baseline"},
    {"id": "file_dev", "type": "file_override"},
    {"id": "rt_dev", "type": "runtime_injection"}
  ],
  "environments": [
    {
      "id": "dev",
      "depth": 0,
      "available_sources": ["base", "file_dev", "rt_dev"],
      "source_priority": ["rt_dev", "file_dev", "base"],
      "bindings": {
        "headroom": {"allowed_source_ids": ["file_dev", "base"]}
      },
      "entries": {
        "base": {"cpu_cores": 4, "headroom": 1.0},
        "file_dev": {"cpu_cores": 6.2505, "headroom": 0.75},
        "rt_dev": {"cpu_cores": 7.0, "headroom": 0.5}
      }
    }
  ],
  "tasks": [
    {"id": "capacity", "formula": "1.5*cpu_cores - 2*headroom"}
  ]
}
```

约束与校验：

- 环境 ID、源 ID、任务 ID 分别唯一；环境 `depth` 是非负整数。
- 源类型只支持 `baseline`、`file_override`、`runtime_injection`，类型偏置必须完整声明且为非负整数。
- 数值仅接受有限整数、浮点数或 JSON 十进制数；布尔值和数字字符串会被拒绝。
- `available_sources` 必须引用已声明源；`source_priority` 必须与可用源集合完全一致。
- 公式只支持加、减、数值常量乘法和括号，必须至少引用一个数值键，且引用键至少存在于一个可用源中。拒绝除法和键与键相乘。
- 绑定只支持 `allowed_source_ids`、`excluded_source_ids`、`allowed_source_types`；引用的源必须存在。
- 任一加载失败都会整体拒绝，不产生部分环境结论。报告包含业务码、环境和 JSON 风格字段路径；`E_ENV`、`E_BIND`、`E_GRID` 同时带 `parent_code: E_SPEC`。

## 裁决与定点语义

对每个环境中的每个键，只把实际提供该键的源放进链，按以下元组升序稳定排序：

```text
(来源类型偏置, 环境显式优先级序号, 来源 ID 字典序)
```

链记录每一步的源 ID、类型、原值、定点值、类型偏置、显式优先级、排序依据、绑定是否满足和是否选中。生效值为链上首个满足逐键绑定约束的源。排序不依赖字典或哈希遍历，所有列表和摘要都使用规范 JSON。

设缩放因子为 `S = 10^scale_power`：

- 所有配置值和公式系数使用 `trunc_toward_zero(value * S)` 量化。
- 系数定点值乘键定点值，整数乘加后对 `S` 做向零截断，最终结果饱和到有符号 64 位整数范围。
- 单键误差报告包含带符号残差、绝对残差与理论上界 `1/S`。
- 任意两个键的比较误差界为两者 `1/S` 之和；差值落入该界时使用键 ID 字典序稳定决胜，并标为受误差界影响。

任务结果误差界拆成三段：

- `source_quantization`：按公式系数绝对值加权的键量化残差。
- `formula_truncation`：系数量化残差、常量量化残差和最终整数除法余项。
- `aggregation_synthesis`：来源与系数量化交叉项，以及最终饱和间隙。

三段贡献之和即紧误差界；报告另给基于统一 `1/S` 的理论上界。主导段在贡献占比 `>= error_dominance_threshold` 时成立，占比相同则按上述三段固定顺序决胜。所有中间项逐键列在 `intermediates` 中。

## 版本、账本与快照

- `environment_version` 覆盖源定义、环境层级、可用源、绑定、源条目和任务公式。
- `priority_version` 覆盖来源类型偏置、源类型、环境可用源集合与显式优先级。
- `scale_version` 覆盖缩放因子和溢出策略。
- 打开已修改声明时，版本立即单调递增，旧当前结论退为非当前；界面当前视图整体清空，不逐键合并。重新计算会创建新的 run，旧 run 及其结论保留为只读历史。
- 每个环境计算完成后，在同一个 SQLite 事务中写入结论载荷、链摘要、格式版本字段和三个版本号。环境记录有禁止 UPDATE/DELETE 的触发器。
- 如果进程被强制终止，`environment_progress` 中未提交的环境会在下次打开后标成 `unknown`；只恢复已经完整提交的环境，明确要求重跑，不伪造强制退出报告。
- 快照记录三个版本、缩放口径、完成环境集合、逐环境结论和内容摘要。版本一致时重新裁决并逐环境逐键逐字段比较；首个差异以 `E_REPLAY` 返回环境、键、字段和两边依据。版本不一致时整体重建当前结论视图，而非合并旧快照。

## 异常注入与日志

三类注入只生成坏输入文件或非法会话参数，不改变程序分支：

```bat
py -3 -m deploy_workbench run --spec data/sample_spec.json --state-root .workbench --inject-env
py -3 -m deploy_workbench run --spec data/sample_spec.json --state-root .workbench --inject-bind
py -3 -m deploy_workbench run --spec data/sample_spec.json --state-root .workbench --inject-grid
```

| 业务码 | 含义 | CLI 退出码 |
| --- | --- | ---: |
| `E_SPEC` | 声明整体校验失败 | 5 |
| `E_ENV` | 环境声明文件损坏 | 2 |
| `E_BIND` | 绑定引用缺失来源 | 3 |
| `E_GRID` | 缩放因子非法 | 4 |
| `E_REPLAY` | 快照重放首个差异 | 6 |
| `E_LEDGER` | 账本格式或摘要损坏 | 7 |

每次失败都会向 `--state-root/logs/workbench.jsonl` 追加一行结构化 JSON，包含业务码、路径、环境、消息和事件类型。

## Tk 界面

界面左侧可增删改环境、源类型和逐源键值条目；右侧标签页稳定展示：

- 环境摘要与提交状态。
- 逐环境逐键来源链和选中依据。
- 生效原值、定点值、量化误差和误差热力。
- 任务结果、误差界及三段归因。
- run/版本列表，并可只读打开历史结论。

“Refresh view (read-only)”只读取账本，不触发生成、恢复或重算。

## 测试与验收

```bat
run_tests.cmd
py -3 scripts/verify_acceptance.py
```

共 29 项 `unittest` 测试覆盖：加载强校验、三类注入、稳定来源链、绑定跳过、定点误差界、归因、比较决胜、版本递增、陈旧隔离、只读历史、强制中断恢复、快照重放和首个差异定位。

实际执行结果与样本数值见 `docs/VERIFICATION.md`，机器可读摘要见 `artifacts/acceptance_report.json`。架构与不变量说明见 `docs/ARCHITECTURE.md`。
