# 实际验证报告

## 1. 执行环境

```text
Python 3.11.0rc1
numpy 2.4.6
PySide6 6.12.0
Linux/WSL x86_64
```

实现不含 Docker 或其他容器依赖。Windows 安装和运行命令见 `docs/RUNNING.md`。

## 2. 自动化测试

执行：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

结果：

```text
Ran 10 tests in 18.823s
OK
```

覆盖内容：

- 场景与种子审计逐字节一致、子种子独立复算。
- 扁平化重复一致、容差收紧时每个对应子路径顶点数单调不减。
- 退化直线跳过、共线三次贝塞尔降级为直线。
- 自相交样本非零与奇偶覆盖率存在差异且首差异位置固定。
- 层像素整体替换会改变层内容哈希。
- 六类参数注入均产生预期业务码且不崩溃。
- 畸形注入操作、非数值坐标/容差也会归类为业务错误而不抛未捕获异常。
- 独立验证器正常样本全部检查通过。
- 像素变异在 `L0000` 的 `(60, 54)`、BGRA 通道 `0` 定位首差异。
- Alpha 越界和像素变异最小反例均可复现且每个保留元素删除后违规消失。
- 两套完整运行逐文件哈希一致。

## 3. 固定演示

执行：

```bash
.venv/bin/python -m vgsynth demo --out artifacts/demo
```

结果：

```text
run_id = b89b2a45c508e6221fed8d7ea5b87338cccc4a1187d1b9113b296968e8a233f6
case_count = 12
verified = 3
verified_with_findings = 7
expected_verifier_mismatch = 2
failed_cases = []
```

双轮比较：

```text
equivalent = true
file_count_first = 564
file_count_second = 564
only_first = []
only_second = []
differing = []
run_manifest_equal = true
```

`comparison.json` SHA-256：

```text
f5b1d1f185449179e0b25fa228b7ddd52f7c23f929c9a564f667c354a3e98b90
```

`run_a/report.json` 与 `run_b/report.json` 的 SHA-256 均为：

```text
81a7585a80c20f6d7b5ba0f45340fe59e2c6b3733b18109bda2641f56f29cdcd
```

## 4. 整轮验证

执行：

```bash
.venv/bin/python -m vgsynth verify --run-dir artifacts/demo/run_a
```

退出码 `0`。结果：

```text
case_count = 12
failed = []
expected_mismatch = 2
```

两个预期不一致：

| 样本 | 类型 | 层 | 首个位置 | 结果 |
|---|---|---|---|---|
| `verifier_pixel_mismatch_injected` | `layer_pixel` | `L0000` | `(60, 54)`，通道 `0` | 期望 `39`，实际 `56` |
| `layer_hash_mismatch_injected` | `layer_hash` | `L0000` | `(0, 0)` | 清单哈希被参数注入确定性替换 |

## 5. 正常样本全检查

`normal_cross`：

```text
status = ok
scene_hash_valid = true
injection_classification_match = true
layer_headers_match = true
layer_hashes_match = true
layer_pixels_match = true
fill_rule_coverage_match = true
final_buffer_match = true
canvas_hash = b75470a40ea2b567e4493cd9712337bb239c38de21d1c7470353c2763ab2bb6a
```

图层哈希：

```text
L0000 = 66895babbeaf14ffe62ed6c0921df6fb96f80d0d8fa2c821448f718b42e973c2
L0001 = 53ac3a15bb6f6a1ab0bcc731b7b68a82e06ee1f71dd068e38512c16bac7f3e34
```

## 6. 双填充规则对照

自相交五角星 `self_intersect_loop` 的 `L0000`：

```text
nonzero_hash = 9efdf6d0fab6f720b2ae0840b023a806208e24121c2a3c9568a58aa40b0131e9
evenodd_hash = bc11bad1a876b7222ca7c6009fedcdb4d468c13fd7faf236be3b0db74cc4fce4
first_difference = (x=104, y=51), count=930
selected_rule = evenodd
```

差异像素同时记录两种规则下的 8 位覆盖率，两个哈希均进入层头并参与最终层哈希。

## 7. 最小反例结果

所有带发现或预期不一致的样本都生成 `minimal_counterexample/counterexample.json`，`status=reproduced`、`irreducible=true`。摘要：

| 样本 | 必要元素 | 反例哈希前缀 |
|---|---|---|
| `alpha_overflow_injected` | path `p0`、subpath `s0`、segment `0`、injection `inj-alpha` | `f0dff44da09131f7` |
| `invalid_fill_rule_injected` | path `p0`、subpath `s0`、segment `0`、injection `inj-rule` | `db00720e0c0f1309` |
| `coordinate_out_of_bounds_injected` | path `p1`、subpath `s0`、segment `0`、injection `inj-coord` | `8795d643af56fff1` |
| `negative_tolerance_injected` | injection `inj-tolerance` | `959d1949bbed0eb2` |
| `zero_stroke_injected` | path `p0`、subpath `s0`、segment `0`、injection `inj-width` | `0b3a138e9a064c02` |
| `injection_conflict_injected` | path `p0`、subpath `s0`、segment `0`、injection `inj-conflict-a` | `43b14748d9422c82` |
| `verifier_pixel_mismatch_injected` | path `p0`、subpath `s0`、segment `0`、injection `inj-pixel` | `bc359c6aadaf501a` |
| `layer_hash_mismatch_injected` | path `p0`、subpath `s0`、segment `0`、injection `inj-layer-hash` | `744388129b13fe3d` |

每个元素都执行了删除测试；删除后 `still_triggers=false`，目标违规不再复现。

## 8. 产物位置

- 双轮比较：`artifacts/demo/comparison.json`
- 第一轮报告：`artifacts/demo/run_a/report.json`
- 第一轮文件哈希：`artifacts/demo/run_a/hashes.json`
- 各层 BGRA/PNG：`artifacts/demo/run_a/cases/<case_id>/layers/`
- 覆盖率：`artifacts/demo/run_a/cases/<case_id>/coverage/`
- 最小反例：`artifacts/demo/run_a/cases/<case_id>/minimal_counterexample/`
