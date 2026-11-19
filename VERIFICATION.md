# 实际验证记录

## 执行环境

- Windows PowerShell `5.1.26100.9818`
- Windows Python `3.14.7`（`C:\Users\27857\AppData\Local\Programs\Python\Python314\python.exe`）
- `numpy 2.5.2`
- `PyYAML 6.0.3`
- 命令由 Windows PowerShell 直接调用 Windows Python；工作区通过 `\\wsl.localhost\Ubuntu-22.04\...` UNC 路径访问
- 未使用 Docker 或其他容器

环境检查命令：

```powershell
python scripts/check_env.py
```

输出：

```text
python=3.14.7
executable=C:\Users\27857\AppData\Local\Programs\Python\Python314\python.exe
numpy=2.5.2
pyyaml=6.0.3
```

## 自动化测试

```powershell
python -m unittest discover -s tests -v
```

结果：`Ran 20 tests in 1.117s`，`OK`。覆盖：

- 正常解析、未知变量/分量、缺阈值、表达式白名单、重复 YAML 键和图环。
- 两次重放的逐位状态、边序哈希和逐步哈希一致。
- 边界裕度、误差区间、不确定标记和逐边系数/舍入上界。
- 扰动扫描的确定性、排名和 `Spec` 不被修改。
- 合流路径首个分歧分量与两条边序列。
- 阈值/系数/初值注入、首次判定翻转和结局变化。
- 七个业务错误码以及快照陈旧检测。
- demo 两次构建逐字节一致。

## 可重复 demo

```powershell
.\run_demo.ps1
```

`artifacts/demo` 共生成 20 个文件；`snapshot.json` 校验其中 19 个非快照文件，全部通过。

两次独立目录运行：

```powershell
python -m narrative_analyzer demo --out-dir artifacts/verify-a
python -m narrative_analyzer demo --out-dir artifacts/verify-b
python scripts/compare_dirs.py artifacts/verify-a artifacts/verify-b
python -m narrative_analyzer snapshot-check --snapshot artifacts/verify-a/snapshot.json
```

结果：

```text
IDENTICAL 20 files
snapshot artifact_count = 19
business_code = OK
```

## 关键验收数值

正常路径结局为 `good`，最终状态向量及 IEEE-754 位模式：

```text
values = [0.30000000000000004, 0.15000000000000002, 0.03]
bits   = [3fd3333333333334, 3fc3333333333334, 3f9eb851eb851eb8]
```

边界判定 `c_trust_high`：

```text
current = 0.30000000000000004
threshold = 0.3
signed_margin = 5.551115123125783e-17
margin_error_upper = 6.00044408920985e-13
reachable = true
boundary = true
uncertain = true
```

分支合并分歧：

```text
component = trust
left  = 0.30000000000000004  via [e_start_a, e_a_join]
right = 0.3                  via [e_start_b, e_b_join]
absolute_difference = 5.551115123125783e-17
path_tolerance = 1e-18
business_code = E_PATH_DIVERGE
```

敏感度排名第一项：

```text
coefficient = e_start_b/effect[0]/coef[0]
relative_delta = 1e-15
condition = c_trust_high
before.reachable = false
after.reachable = true
```

阈值注入首翻：

```text
condition = c_trust_high
threshold: 0.3 -> 0.29999999899999996 (delta=-1e-9)
margin:    0.0 -> 1.0000000272292198e-09
reachable: false -> true
outcome:   bad -> good
```

## 内容哈希

```text
spec.yaml              d20ff70ec2e6bf6d23bdd081dd787b3f819d8c063fc8b7444629f04ab921dbdb
replay_primary.json    ae2f0013bbcec32ecdce34febcf0c40d36db48ac1511f7681af3237327504ee3
paths.json             d6ab5ca56fb40bf8e5c7a9e90bfafd01bae4969186ab683b7f3ea8687fa0766c
sensitivity.json       5bc4212751d056341b1c851f9d25118fa1b61680888f6a3a4b89ddf0de9afc8e
replay_injected.json   e8167c93cc075c198d7536e3b335ff0c8ab4d4be945ca502a92da4f5a1f79f07
demo_summary.json      b10896376cfcc005dd44d7a1267f5fc8fcd8a09dd7fbd8a632f95961767c6822
snapshot.json          e6e47a70b85a3059a12fc0ac1899a881592d117ea995cbf0171ccb286a820526
```

逐运行目录比较同时校验了全部 20 个文件的 SHA-256；两个目录的 `demo_summary.json` 和 `snapshot.json` 也分别完全一致。

## 异常链路

`artifacts/demo/error_cases.json` 中七个业务码均按固定顺序真实触发：

```text
E_SPEC_INVALID
E_GRAPH_CYCLE
E_SELECTION_UNKNOWN
E_PRECISION_BOUND
E_PATH_DIVERGE
E_INJECT_CONFIG
E_SNAPSHOT_STALE
```

