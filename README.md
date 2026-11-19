# 叙事分支与事件因果工作台

一个完全本地、确定性的桌面工作台，用于推进 YAML 叙事分支、累积角色状态向量、核对汇合路径一致性、记录事件因果链，并对系数/阈值修改做精确失效闭包与注入翻转定位。

项目不依赖网络、外部素材、数据库或容器。YAML/JSON 全部使用 UTF-8；样例由固定整数种子生成；运行时不读取墙钟时间。

## 快速开始（Windows）

要求 Python 3.11+。Windows 官方安装包默认带 Tkinter。

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m tkinter
```

生成固定样本、执行完整演示、输出逐节点哈希与结构化报告：

```bat
python -m story_workbench demo --out artifacts\demo
```

也可以直接双击/运行 `run_demo.bat`。启动桌面时间轴：

```bat
python -m story_workbench gui --spec artifacts\demo\samples\normal.yaml --sequence normal_path
```

运行测试：

```bat
python -m unittest discover -s tests -t . -v
```

## 命令

| 命令 | 作用 |
|---|---|
| `python -m story_workbench demo --out artifacts/demo` | 生成四类固定样本并执行推进、裁决、多路径核对、增量失效和注入定位；同一输入连续构建两次并逐字节比较 |
| `python -m story_workbench generate-samples --out samples` | 只生成固定样本与注入配置 |
| `python -m story_workbench validate SPEC.yaml` | 解析、校验并输出确定性规格序列化与规格哈希 |
| `python -m story_workbench run SPEC.yaml --sequence ID --out run.json` | 重放一条选择序列，输出快照、状态、因果链、裕度和哈希 |
| `python -m story_workbench inject SPEC.yaml --config CONFIG.json --sequence ID --out flip.json` | 应用参数注入并定位首个翻转判定 |
| `python -m story_workbench gui --spec SPEC.yaml --sequence ID` | 打开 Tkinter 时间轴工作台 |

`demo` 的默认输出：

- `artifacts/demo/samples/`：`normal.yaml`、`merge.yaml`、`boundary.yaml`、`injection.yaml` 及注入配置。
- `artifacts/demo/report.json`：完整结构化报告，含所有快照、路径轨迹、因果链、闭包和复现记录。
- `artifacts/demo/node_hashes.json`：规格、时间轴快照、依赖图节点、汇合节点与注入报告的哈希清单。
- `artifacts/demo/report.md`：适合人工审阅的摘要。

## 固定样本

- `normal.yaml`：正常推进；同一边上重复写入同一分量触发最后写者裁决；节点多条出边的潜在叠加由投影账本记录。
- `merge.yaml`：`m_ok` 两条路径产生 `5.551115123125783e-17` 的微小差异，落在 `1e-9` 容差内；`m_bad` 两条路径首个分歧分量为 `trust`，差值为 `0.09999999999999998`。
- `boundary.yaml`：`0.1 + 0.2` 得到 `0.30000000000000004`，对阈值 `0.3` 的判定被标为 `boundary`，而不是直接判为可达/不可达。
- `injection.yaml`：基线结局 `good`；阈值注入后所选边不可达，结局翻转为 `blocked:i_gate`，首个翻转判定为 `c_trust_good`。

## 确定性语义

### 表达式

效果表达式只允许数字、状态分量标识符、括号、加法 `+` 与乘法 `*`。二元减法、除法、幂、函数和属性访问均返回 `E_SPEC_INVALID`。一元正负号可作为数字符号使用。

```yaml
effects:
  - id: f_fear
    target: fear
    expr: '0.05 * (trust + 1.0)'
```

表达式 RHS 的结果被解释为该分量的增量。所有 RHS 都针对边执行前的不可变状态求值；乘法优先于加法，同级从左到右，括号显式改变结合顺序。每一步均使用 IEEE-754 binary64；安装 NumPy 时显式经 `numpy.float64` 归一，缺失时等价回退到 Python `float`（同为 binary64）。

### 冲突裁决

同一条边上的多个效果按效果列表声明顺序执行“分量级最后写者”裁决：

1. 所有效果先针对边执行前状态求值。
2. 对每个目标分量，声明顺序最后的写入者生效。
3. 被覆盖效果、覆盖原因、候选值与最终生效值写入裁决账本。
4. 仅生效效果按声明顺序写入状态；每个分量追加一条因果贡献。

YAML 映射键顺序不影响结果；列表顺序具有语义。正常时间轴中一个选择只走一条出边。若把某节点的全部出边效果作为并发组叠加，`outgoing_effect_projection` 使用“边声明顺序 + 效果声明顺序 + 分量级最后写者”形成可复算投影；它不会错误地把互斥分支写入实际状态。

### 裕度与边界

对阈值 `t` 和当前值 `v`：

- `signed_margin = v - t`，同时输出按比较方向定义的 `directed_margin`。
- 比较方向由 `op` 的 `<`、`<=`、`>`、`>=` 明确给出。
- `abs(signed_margin) < boundary_tolerance` 时判定状态为 `boundary`。
- `boundary` 分支带 `reachable: false`，但不会被归类为普通的硬失败；边界清单包含版本戳、内容哈希与逐条件内容。

### 因果链

每个状态分量保存初值、最终值以及按执行顺序排列的贡献：

```text
edge_id / effect_id / before / delta / after
```

因此可回答“当前值来自哪些边”。被最后写者覆盖的效果保存在裁决账本中，但不会进入数值因果链。

## 依赖图与失效闭包

每条边的 `trigger_keys` 显式包含：

- `effect_set:<edge_id>`
- `<edge_id>:<effect_id>:coef:<ordinal>`
- `<edge_id>:<condition_id>:threshold`

依赖图节点类别包括 `edge_effect`、`raw_state`、`path_state`、`condition`、`ending`。修改单个系数或阈值时：

1. 从触发键取得所有者节点。
2. 沿反向依赖边求精确下游闭包。
3. 仅按拓扑顺序重建闭包内节点。
4. 闭包外依赖直接复用缓存版本与内容哈希。
5. 断言实际重算节点集合与闭包集合完全相等。

报告记录闭包、实际重算集合、复用来源、每节点版本戳与前后哈希。演示中的系数修改闭包为 9 个节点，阈值修改闭包为 5 个节点，闭包外哈希均保持不变。

## 注入格式

注入配置是可序列化、可随运行记录复现的 JSON。支持四种操作：

```json
{
  "schema": "narrative-workbench/injections/v1",
  "id": "example",
  "operations": [
    {"kind": "coefficient_perturb", "edge_id": "e1", "effect_id": "f1", "coefficient_index": 0, "mode": "add", "value": 0.05},
    {"kind": "threshold_shift", "edge_id": "e2", "condition_id": "c1", "delta": 0.11},
    {"kind": "initial_offset", "component": "fear", "delta": 0.01},
    {"kind": "remove_effects", "edge_id": "e3"}
  ]
}
```

注入只作用于输入的规格/参数副本，不修改源规格对象。若结局变化，按实际时间轴顺序、边序号和条件序号定位首个判定翻转，输出前后裕度、裕度区间是否跨零、注入片段与完整注入记录。

## 桌面工作台

GUI 只读取 `TimelineResult` 中不可变的 `Snapshot` 对象，不重新计算状态。每个快照展示：

- 状态版本戳、内容哈希、状态哈希、节点与结局。
- 按规格声明顺序排列的状态分量与逐边因果来源。
- 全部后续分支、条件当前值、阈值、有符号裕度和边界标记。
- 实际边效果裁决账本与同节点出边投影账本。

当前交付容器未安装 Tk 运行库且无 root 权限，因此只能验证 Tkinter 不可用时的结构化错误 `E_GUI_UNAVAILABLE`；Windows 官方 Python 环境按上述命令可直接启动。核心引擎、CLI、报告与测试不依赖 Tk。

## 实际验证结果

本工作区已执行：

- `python -m unittest discover -s tests -t . -v`：29 项测试全部通过。
- `python -m story_workbench demo --out artifacts/demo`：两次内存构建 `byte_identical=true`。
- 连续两个独立 CLI 进程重写同一输出目录：`report.json`、`node_hashes.json`、`report.md` 逐字节一致。
- `report.json` SHA-256：`1200b39c11ccc04d7613c5853fc9fc6ec9514a38ccf3256907d24d97288498d2`
- `node_hashes.json` SHA-256：`b6658210fd4e3c022d3b41aa5489b63093be8237908c5d2a98964092284608fe`
- `report.md` SHA-256：`4416b40a0f8b8a7cd082d5cfefaf63ad3f8451ba64285cd3987c35e5fe6ed87a`
- 正常路线最终状态：`trust=0.30000000000000004`、`reputation=0.5`、`fear=0.1`；边序哈希 `e064530b2c8252a0…`，重放哈希 `44d3be6c3c17668c…`。
- 注入翻转：`good -> blocked:i_gate`；`c_trust_good` 裕度 `0.09999999999999998 -> -0.010000000000000009`。

更细的模块边界、哈希定义和验证方法见 `docs/TECHNICAL.md`。
