# GlyphForge

GlyphForge 是一个面向字体轮廓、扫描线光栅化和灰度字形图集的本地桌面应用/领域库。全部样本由程序构造，不读取系统字体文件、不访问网络；同一条固定编辑序列可重复生成逐字节一致的位图哈希、图集账本和线程预算报告。

## 快速开始（Windows）

要求 Python 3.10+，推荐 Python 3.11。依赖只有 NumPy、PyQt5 和测试用 pytest。

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 固定 3 个工作线程运行演示；内部会执行两次序列并比较字节
python -m glyphforge demo --out artifacts\demo --threads 3

# 打开只读诊断视图 + 外部编辑控制器
python -m glyphforge gui
```

也可以直接运行 `powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1`。设置脚本会创建 `.venv`、安装依赖、运行测试并生成演示产物；演示启动时先写入并删除 `.write-probe`，因此输出目录不可写会立即失败。

如果 PowerShell 禁止激活脚本，可直接使用 `.venv\Scripts\python.exe`，无需修改系统策略。

## 已实现能力

- 不可变字形模型：闭合轮廓、直线/二次/三次贝塞尔段、字形 ID、建议尺寸档位；规范化 JSON、有限坐标及结构校验。
- 自适应曲线扁平化：按尺寸推导容差，递归 De Casteljau 细分，每个曲线段记录误差估计与细分数。
- 扫描线覆盖光栅化：每像素 4 条水平子扫描线，解析计算水平覆盖率并累积为 8 位灰度；支持非零环绕和奇偶规则。
- 并行调度：固定线程池、任务键规范化归并、线程数预算与线程内存总预算；预算不足时降并行度并记录原因。
- 固定 512×512 灰度图集：最佳短边空闲矩形分配、区域映射、空闲矩形账本、原子复用、页面回收与重打包。
- 可复现降级：页面回收 → 重试原任务 → 按尺寸从高到低尝试 fallback → 延迟打包；每项决策都有单调序号。
- 增量重算：控制点、填充规则和尺寸档位变更只重算对应 `(glyph,size,fill_rule)` 键；变更闭包来自真实光栅结果。
- 不可变诊断快照：位图、图集布局、空闲/复用标记、线程预算、降级记录一次整体替换；视图不持有应用服务。
- 业务错误码、完整异常链、程序化正常/退化/密集样本、JSON+PGM 演示账本和双运行字节一致性检查。

## 数据格式

轮廓首点由 `start` 给出，后续段隐式从前一段的 `to` 连到下一段，因此不存在相邻段位置断裂。`closed` 必须明确为 `true`；最后一个端点回到首点的闭合边由光栅器添加，也可以显式写一段回边。

```json
{
  "format": "glyphforge-glyph-v1",
  "glyphs": [{
    "id": "A",
    "sizes": [12, 16, 24, 32, 48],
    "contours": [{
      "closed": true,
      "start": {"x": 90.0, "y": 40.0},
      "segments": [
        {"type": "line", "to": {"x": 390.0, "y": 920.0}},
        {"type": "quadratic", "control1": {"x": 520.0, "y": 920.0}, "to": {"x": 620.0, "y": 920.0}},
        {"type": "cubic", "control1": {"x": 780.0, "y": 470.0}, "control2": {"x": 830.0, "y": 300.0}, "to": {"x": 660.0, "y": 120.0}}
      ]
    }]
  }]
}
```

规范化序列化使用 UTF-8、LF、键排序、紧凑分隔符、有限浮点数；字形的输出顺序按 ID 排序。`loads_glyphs` / `dumps_glyphs` 可完成稳定往返：

```powershell
python -m glyphforge validate artifacts\demo\samples.json
```

## 关键约定

**尺寸与容差。** 坐标单位为 1000 units/em。`flatten_tolerance_for_size(size)` 将默认 0.35 像素的目标误差换算为该尺寸下的字形单位容差：`tolerance_units = 0.35 * units_per_em / size`。因此更大尺寸档位对应更小的单位容差；质量参数可进一步收紧。

**扫描线。** 每个输出像素行使用 4 条等距水平子扫描线，半开区间 `[min_y,max_y)` 避免水平边和顶点重复求交。交点按 `(x,direction,edge_order)` 稳定排序。奇偶规则两两配对；非零规则按 `+1/-1` 有向环绕累计，在 0↔非 0 边界取水平 span。span 与像素列做解析重叠，故边缘得到 0–255 灰度。`scanline_samples=1` 时采样点就是像素中心 `(row+0.5)`。

**内容哈希。** `bitmap_hash` 是 `SHA-256("GLYPHFORGE-BITMAP\0" + width:u32be + height:u32be + grayscale bytes)`。哈希包含宽高，避免相同字节序列的不同形状碰撞。

**线程预算。** 单任务内存估算由最大像素方框、曲线细分上限和交点缓冲组成。有效线程数为 `min(请求线程, floor(总内存/最大任务估算), 任务数)`。最大单任务都不足时会抛出 `E_BUDGET_DENIED`，而不是冒险超额执行。结果按规范化任务键排序，完成顺序不写入产物。

**图集降级。** 图集每页默认 512×512，区域和空闲矩形都持久化到账本。预算不足时固定执行：回收完全无引用的页 → 重试原请求 → 尝试同字形、同规则、较低尺寸档位的缓存位图 → 延迟打包。尝试顺序、报价和序号只依赖输入状态。已发布且未受本次编辑影响的区域会记录坐标/内容哈希/像素快照，提交后逐项校验，防止并发或重排覆盖。

**增量闭包。** 每次编辑先形成不可变的新字形状态，在工作池完成全部受影响任务后才提交。报告中的 `recomputed_glyph_ids` 只由实际返回的光栅结果键生成，不包含仅重打包或无变化 glyph，因此与实际重算集合精确一致。

**快照。** `DiagnosticSnapshot` 只含 frozen dataclass、tuple、`bytes` 和只读 NumPy 视图。`DiagnosticViewState.apply` 对同版本返回 `false`，对旧版本抛 `E_SNAPSHOT_STALE`，对新版本整体替换。PyQt5 的 `DiagnosticView` 没有任何应用会话引用或重算方法；编辑按钮位于 `EditorWindow`，通过外部回调提交命令。

## 演示序列

`python -m glyphforge demo` 构造 7 个样本字形：

- `A`、`O`、`B`：直线/二次/三次曲线、孔洞和不对称轮廓；
- `i`：分离的笔画与点轮廓；
- `overlap`：同向嵌套轮廓，用它验证非零与奇偶规则产生不同像素；
- `degenerate`：所有控制点退化到同一点，验证全零位图；
- `dense`：63 个细小矩形轮廓，验证密集小字和任务内存压力。

固定步骤：

1. 31 个多字形、多尺寸任务并行光栅化并打包；刻意启用较小图集/内存预算，产生 fallback、延迟和线程降级记录。
2. 修改 `A` 的二次控制点，闭包必须只有 `A`，其他字形位图和图集区域不变。
3. 把 `overlap` 切到 `evenodd`，闭包只有 `overlap`，并验证与 `nonzero` 哈希不同。
4. 把 `dense` 切到 24px，闭包只有 `dense`。
5. 取消发布并回收空页。
6. 重新打包 `A@48`，逐像素哈希必须与回收前相同。

输出目录：

```text
artifacts/demo/
  ledger.json             # 六步完整报告、闭包、降级决策和最终快照
  samples.json            # 规范化程序化样本
  thread_budget.json      # 首次并行渲染的线程预算
  atlas_ledger_final.json # 最终页、区域、空闲矩形和复用信息
  snapshot_final.json     # 最终不可变诊断快照（不含原始像素）
  atlas_pages/page-*.pgm  # 8 位灰度 PGM 图集页
  determinism.json        # 双运行哈希与全部不变量
  run_manifest.json       # 输出文件大小与 SHA-256
```

## 错误码

| 错误码 | 典型触发 |
|---|---|
| `E_OUTLINE_INVALID` | JSON 非法、轮廓未闭合、段类型非法、控制点缺失/非有限/越界 |
| `E_FILL_RULE` | 未知填充规则 |
| `E_TASK_KEY` | 重复任务、非法尺寸、未知字形/任务、结果缺失 |
| `E_ATLAS_FULL` | 页数、占用像素或单页尺寸不足 |
| `E_BUDGET_DENIED` | 线程预算非法、单任务超过总内存、扫描线工作缓冲超限 |
| `E_SNAPSHOT_STALE` | 视图以旧版本覆盖新快照或读取过期快照 |
| `E_BITMAP_HASH` | 期望哈希不符、未受影响位图/区域在提交后变化 |

`GlyphForgeError.to_dict()` 提供稳定的 `code/message/glyph_id/segment_index/details`，可直接进入诊断账本。

## 代码结构

```text
glyphforge/model.py        不可变模型、校验、规范化 JSON
glyphforge/geometry.py     容差推导、贝塞尔自适应扁平化
glyphforge/raster.py       扫描线填充、灰度覆盖、哈希
glyphforge/scheduler.py    固定线程池、任务归并、线程内存预算
glyphforge/atlas.py        固定页分配、空闲矩形、降级协调器
glyphforge/application.py  编辑事务、增量闭包、区域不变性校验
glyphforge/diagnostics.py  不可变版本快照与视图状态
glyphforge/gui.py          PyQt5 只读诊断视图和外部编辑控制器
glyphforge/samples.py      完全程序化样本
glyphforge/demo.py         固定序列、双运行校验、PGM/JSON 账本
tests/                     33 个单元/集成/端到端测试
```

## 验证结果

当前工作区实际执行环境为 Linux、Python 3.11.0rc1、NumPy 2.4.6、PyQt5 5.15.14、pytest 9.2.0；代码只使用跨平台 Python/NumPy/PyQt5 API，并提供 Windows PowerShell 启动脚本。当前环境没有 Windows 主机，因此 Windows 脚本未在本机实机执行；这是本报告唯一的平台验证限制。

```text
$ python -m pytest -q
33 passed

$ ruff check glyphforge tests
All checks passed!

$ python -m glyphforge demo --out artifacts/demo --threads 3
6 steps, two-run SHA-256 27ff402cff4e98bf683d6d78aa960486660035692a15106d036eb6a64c19c4f2,
all 16 demo invariants true
```

最终可核对哈希以 `artifacts/demo/determinism.json` 和 `run_manifest.json` 为准。独立两次 CLI 运行到不同目录时，所有相对路径文件应具有相同 SHA-256；`tests/test_demo.py` 对此进行了自动化断言。
