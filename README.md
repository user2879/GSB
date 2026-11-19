# 叙事分支与事件因果离线分析器

一个不依赖容器、不访问外部服务的确定性分析工具。它读取 YAML 叙事图，在固定选择序列下按声明顺序累积 IEEE-754 双精度状态，输出分支可达性、条件裕度、误差区间、扰动敏感度、分支合流差异、注入首翻定位和内容哈希。

## 环境

- Windows 10/11 + PowerShell
- Python 3.11 或更高版本
- `numpy`、`PyYAML`（只使用本地已安装版本，不下载样例数据）

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install -r requirements.txt
```

不需要 Docker、WSL、数据库或网络账号。项目内的样本、配置和输出全部由 `narrative_analyzer.samplegen` 程序化生成。

## 快速运行

```powershell
python -m narrative_analyzer demo --out-dir artifacts/demo
python -m narrative_analyzer --help
.\verify.ps1
```

`demo` 会在同一目录生成可复现样本、重放报告、敏感度排名、路径核对、注入报告、错误链路、双跑一致性清单和 `snapshot.json`。所有 JSON/YAML 都使用固定键序、`\n` 换行和稳定浮点表示；同一版本连续运行两次逐字节相同。

## CLI

```text
python -m narrative_analyzer validate   --spec SPEC [--output OUT]
python -m narrative_analyzer replay     --spec SPEC --sequence SEQ [--inject INJECT] [--output OUT]
python -m narrative_analyzer sensitivity --spec SPEC --sequence SEQ [--config SCAN] [--output OUT]
python -m narrative_analyzer paths      --spec SPEC [--strict] [--max-paths 1024] [--output OUT]
python -m narrative_analyzer snapshot-check --snapshot SNAPSHOT
python -m narrative_analyzer demo       --out-dir artifacts/demo
python -m narrative_analyzer error-codes
```

业务异常输出为 JSON，包含 `business_code`、`message` 和 `details`。退出码见文末。

## 输入格式

完整样本由 `artifacts/demo/spec.yaml` 生成。核心结构如下：

```yaml
spec_version: 1
seed: 20261119
start: start
comparison: strict_greater
boundary_epsilon: 1.0e-9
path_tolerance: 1.0e-18
numeric:
  coefficient_relative_error: 1.0e-12
  initial_relative_error: 1.0e-12
  unit_roundoff: 1.1102230246251565e-16
  max_abs_error: 1.0e-3
state:
  components: [trust, reputation, fear]
  initial: {trust: 0.0, reputation: 0.1, fear: 0.0}
conditions:
  - {id: c_trust_high, component: trust, threshold: 0.3, direction: ">"}
nodes:
  - {id: start, type: scene}
  - {id: merge, type: choice}
  - {id: ending_good, type: ending, outcome: good}
  - {id: ending_bad, type: ending, outcome: bad}
edges:
  - id: e_good
    source: merge
    target: ending_good
    choice: continue
    when: [c_trust_high]
    effects: []
  - id: e_bad
    source: merge
    target: ending_bad
    choice: continue
    effects: []
```

效果字符串只允许单条赋值语句：`x = expr`、`x += expr`、`x *= expr`。`expr` 的 AST 白名单只包括状态分量、有限数值常量、`+`、`*` 和一元正负号；二元减法、除法、幂、函数调用和属性访问均拒绝。状态分量顺序必须由 `state.components` 显式给出，不能依赖 YAML 映射顺序。

选择序列是列表，或包含 `version: 1` 与 `selections` 的映射：

```yaml
version: 1
selections: [path_a, join, continue]
```

同一节点可以给多条边相同的 `choice`，用于条件路由。引擎按边声明顺序选择首个满足前置条件的候选；因此固定选择序列在参数注入后仍能走向不同结局。一个 `choice` 最多只能有一条无条件候选边，避免不可判定的无条件重复。

## 输出要点

- `replay_*.json`：选择序列、每个判定、每步前后状态、状态位模式、逐步内容哈希、边序哈希、最终结局。
- `paths.json`：所有合流节点的路径状态、逐分量差异、逐判定比较，以及首个分歧分量和两条边序列。
- `sensitivity.json`：每个效果系数的扫描结果、每个判定首次翻转的临界相对扰动和升序排名。
- `replay_injected.json`：注入配置、标准化偏移、基线/注入运行、结局变化和首个布尔翻转判定。
- `error_cases.json`：七个业务错误码的真实触发结果。
- `reproducibility.json`：两次内存构建的逐文件哈希和清单哈希。
- `snapshot.json`：输入与输出文件的 SHA-256；任何依赖文件变化都会触发 `E_SNAPSHOT_STALE`。

## 业务错误码

| 代码 | 退出码 | 触发场景 |
|---|---:|---|
| `E_SPEC_INVALID` | 2 | YAML/字段/表达式白名单/未知分量/未知变量/缺阈值等 |
| `E_GRAPH_CYCLE` | 3 | 节点图含环 |
| `E_SELECTION_UNKNOWN` | 4 | 选择不存在、不可达、提前结束或结束后仍有选择 |
| `E_PRECISION_BOUND` | 5 | 非有限数值或误差上界超过 `numeric.max_abs_error` |
| `E_PATH_DIVERGE` | 6 | 合流路径状态差超过 `path_tolerance` |
| `E_INJECT_CONFIG` | 7 | 注入配置格式错误、未知目标或重复目标 |
| `E_SNAPSHOT_STALE` | 8 | 快照内容哈希、引擎版本或依赖文件已变化 |

## 验证

```powershell
python -m unittest discover -s tests -v
python -m narrative_analyzer demo --out-dir artifacts/demo
python -m narrative_analyzer snapshot-check --snapshot artifacts/demo/snapshot.json
python scripts/compare_dirs.py artifacts/verify-a artifacts/verify-b
```

设计细节、误差公式和确定性约定见 `TECHNICAL.md`；实际执行结果见 `VERIFICATION.md`。

