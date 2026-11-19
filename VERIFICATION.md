# GlyphForge 实际验证记录

验证日期：2026-11-19

## 环境

- Python：3.11.0rc1（当前复核环境）
- NumPy：1.26.4（`pip install numpy==1.26.4`）
- 平台：无 GPU、无显示设备的 Linux/WSL CPU 环境
- 外部输入：无。仅使用固定种子 `20261119` 的本地合成样本
- Windows 兼容性：代码仅使用 Python 标准库与 NumPy，CLI/测试命令均提供 Windows 路径形式；本次未在原生 Windows 主机执行

## 测试结果

命令：

```bash
python3 -m unittest discover -s tests -v
```

结果：

```text
Ran 23 tests in 39.049s
OK
```

覆盖范围：

- 固定种子样本两次生成逐字节一致；
- 五类样本与 12/16/24/32/48 五档尺寸齐全；
- 轮廓闭合、段类型、控制点数量、有限坐标校验；
- 扁平化重复一致、容差收紧顶点数单调不减、退化段显式标记；
- 四类布尔运算面积、重合边消解、不同消解顺序边界哈希一致；
- 9 种端帽/拐角组合的确定性、圆头弧误差和业务错误码；
- 精确像素覆盖、孔洞扣除、首个分歧像素诊断；
- 位图差为正但几何在容差内时不误报漂移，几何越界时返回 `E_CONSISTENCY_DRIFT`；
- 增量闭包、闭包外字形哈希不变、增量结果与全量重建一致；
- 完整演示两次运行逐字节一致。

## 演示报告

生成命令：

```bash
python3 -m glyphforge demo --report reports/demo-report.json
```

两次独立运行的 stdout 与落盘报告均通过 `cmp`：

```text
59f48bd010f562f4804b1bbded7f975ac63cb1723c156a881b7353600102a153  reports/demo-report.json
59f48bd010f562f4804b1bbded7f975ac63cb1723c156a881b7353600102a153  run-2 stdout
59f48bd010f562f4804b1bbded7f975ac63cb1723c156a881b7353600102a153  run-2 report
```

报告大小：148,990 字节。

样本规范序列化哈希：

```text
a1c629b7dd61b6ae663019db83b4de157c823c960ecbf742800c55aa9411ecf2
```

## 双形态一致性

- 布尔结果：`5 字形 × 5 尺寸 × 4 运算 = 100` 组；
- 笔画结果：`5 字形 × 5 尺寸 = 25` 组；
- 边界重光栅化 vs 直接结构光栅化：125 组全部逐像素一致；
- 不一致像素总数：0；
- 每组均保存边界哈希、边界位图哈希、直接位图哈希和一致性计数。

## 容差与消解顺序敏感性

演示比较 `tight(0.5×)`、`loose(2.0×)` 与 `source_then_lex` 顺序，覆盖三个尺寸、并集与笔画：

```text
metrics: 90
E_CONSISTENCY_DRIFT: 0
WITHIN_TOLERANCE_BITMAP_DIFFERENCE: 22
coverage_field fallback: 6 / 90
maximum symmetric-difference area: 40.592929104
maximum Hausdorff distance: 2.050252531262
maximum bitmap differing pixels: 70
```

6 次回退均对应独立扁平化产生的近重合薄片；报告明确记录 `method=coverage_field_integral` 与误差界。22 组位图存在非零像素差但几何在声明容差内，因此没有误报 `E_CONSISTENCY_DRIFT`。

## 增量闭包

初始任务数：330。

| 编辑 | 闭包任务数 | 实际 = 闭包 | 闭包外字形哈希不变 | 增量 = 全量 |
|---|---:|---|---|---|
| 修改 `G01_SIMPLE` 控制点 | 46 | 是 | 是 | 是 |
| 修改 `G04_OPEN_PATH` 端帽/拐角 | 11 | 是 | 是 | 是 |
| 收紧 `G03_OVERLAP_COINCIDENT` 容差至 0.75× | 66 | 是 | 是 | 是 |

最终字形摘要哈希：

```text
G01_SIMPLE              ee035de279ef481f49853c821b8e7f669f749b042588e874f8cc989fdfa65943
G02_SELF_INTERSECT      2c1d0f9e1539626fcd3a245020ce1f20b36ba9e46e2f38801ab42a6b76dcbfa8
G03_OVERLAP_COINCIDENT  bab5703d6175c35989bf363a3b2016a13988248900dd14486bfd87973a6ba740
G04_OPEN_PATH           1022801820b6cf0748c492deec06478df51118ee674a843039fe9f390a42c753
G05_DEGENERATE          99230b7b69da18576b1893d0a3f6fda84023ee1aaf7736ba46d85c9e70baf7ce
```

五个字形的增量摘要均与独立全量重建相同。

## 错误码自检

实际捕获并通过断言：

- `E_OUTLINE_INVALID`：控制点数量不匹配；
- `E_STROKE_WIDTH`：宽度为 0；
- `E_CAP_JOIN`：非法端帽类型。

演示与测试同时覆盖：

- `E_BOOLEAN_DEGENERATE`：边界重建/开放链检查；
- `E_CONSISTENCY_DRIFT`：几何误差超过声明区间；
- `E_CLOSURE_MISMATCH`：实际重算集合或全量对照不一致；
- `E_BITMAP_HASH`：边界与直接位图哈希不一致。

## 复核入口

```bash
python3 -m unittest discover -s tests -v
python3 -m glyphforge demo --report reports/demo-report.json
sha256sum reports/demo-report.json
```

Windows PowerShell：

```powershell
python -m unittest discover -s tests -v
python -m glyphforge demo --report reports\demo-report.json
Get-FileHash reports\demo-report.json -Algorithm SHA256
```
