# NPR 线稿与风格化渲染编辑器

这是一个从零实现的确定性 NPR（Non-Photorealistic Rendering）领域数据编辑与操作应用。输入完全由应用内固定种子生成器产出，不读取外部图片、模型、索引缓存或网络服务。

核心能力：

- NPR 项目 JSON（schema v3）与二进制描边骨架（`NPRSKEL`）双向转换。
- v1/v2 项目固定迁移表、字段补齐与幂等迁移。
- 自适应曲线扁平化、共享描边骨架、逐段误差估计。
- 200 / 2000 / 20000 三档场景的独立光栅化策略选择与预算账本。
- 纯轮廓 `outline`、素描 `sketch`、卡通 `cartoon` 三种模式共享同一份扁平化几何。
- 单控制点、线宽、风格参数编辑的精确失效闭包与逐图元/逐图层缓存复用。
- 不可变带版本渲染快照、陈旧版本拒绝、软件 BGRA 光栅化与可选隐藏 OpenGL FBO 输出。
- 临时文件 + 原子改名、代际目录、提交指针、强制终止恢复与孤儿缓存记录。

## Windows 环境

目标运行环境：

- Windows 10/11 与 Python 3.11 x64。
- 建议安装 NumPy、PyOpenGL、PyOpenGL_accelerate 与本地显卡驱动。
- 不需要 Docker、WSL、外部模型或媒体资产。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[gl]"
python -m npr_edit gl-check
```

`gl-check` 成功时输出隐藏 OpenGL 上下文信息。无法创建上下文时进程不会崩溃，而是输出：

```json
{"error":"E_GL_CONTEXT","message":"...","details":{...}}
```

并返回退出码 `30`。演示默认在上下文不可用时降级到确定性软件 BGRA 光栅化，并把降级写入 `degradation_ledger`；使用 `--require-gl` 可让演示直接以 `E_GL_CONTEXT` 失败。

应用核心算法不依赖 NumPy；当前无 OpenGL 的验证环境也能完整运行全部业务语义。

## 一条命令演示

```powershell
python -m npr_edit demo --workspace .\npr-workspace --seed 20261120 --viewport 256x192
```

固定序列依次执行：

1. 生成 200、2000、20000 三档场景，二次生成逐字节比对，并写出每图元内容哈希清单。
2. 三档场景转二进制骨架再转回 JSON，声明控制点/线宽容差内复原；小档额外验证大端、2 字节索引、`u16_fixed` 线宽。
3. 对缺失风格参数、描边采样与图层顺序的 v1 项目执行迁移，再迁移一次校验幂等。
4. 为三档选择光栅化策略并记录预算、预估耗时、内存、误差与选择依据。
5. 三档全量渲染并执行单图元增量编辑；小档继续切换 `sketch`、`cartoon`，验证几何未重建。
6. 持久化 2000 档，启动真实子进程写入未提交目录后 `os._exit(137)`，父进程从最近 `CURRENT` 代际恢复。
7. 基于恢复出的版本戳继续增量编辑，最终正常关闭，再恢复验证 `CLOSED` 标记。

标准输出是规范化 JSON。完整报告同时写到：

```text
npr-workspace/report/summary.json
```

主要产物：

```text
npr-workspace/generated/scene-{tier}.json
npr-workspace/generated/element-hashes-{tier}.json
npr-workspace/generated/scene-{tier}.nprb
npr-workspace/rendered/tier-{tier}.bgra
npr-workspace/rendered/tier-{tier}.bmp
npr-workspace/rendered/ledger-{tier}.json
npr-workspace/state/...
```

`ledger-{tier}.json` 包含逐图元骨架哈希、渲染哈希、几何哈希与逐图层内容哈希。

## 两次运行逐字节验证

```powershell
python -m npr_edit verify --workspace .\npr-verify --seed 20261120 --viewport 256x192
```

该命令分别在 `run-a`、`run-b` 执行完整固定序列，然后比较：

- 两次规范化摘要的 SHA-256。
- `generated/`、`rendered/`、`state/` 与 `report/summary.json` 的每一个文件哈希。
- 两次运行的全部业务验证位。

结果写入 `npr-verify/report/verification.json`。任何摘要或产物不一致都会返回退出码 `4`。

## 开发与测试

```powershell
python -m unittest discover -s tests -v
```

测试覆盖生成确定性、扁平化确定性/单调性、退化段、迁移幂等、大小端与索引宽度、线宽编码容差、三档策略、增量闭包、模式切换共享几何、快照陈旧错误、原子提交、孤儿回收、正常关闭和 `E_GL_CONTEXT`。

## NPR JSON 项目格式

当前 `schema_version` 为 `3`。根字段：

```json
{
  "schema_version": 3,
  "generator": {"kind": "npr-deterministic-hash-v1", "seed": 20261120, "tier": 200},
  "canvas": {"width": 640, "height": 480},
  "sampling": {
    "mode": "adaptive",
    "tolerance_px": 0.45,
    "min_subdivisions": 1,
    "max_subdivisions": 18,
    "width_scale": 1.0
  },
  "layers": [
    {"id": "L00", "name": "background-ink", "order": 0, "visible": true}
  ],
  "elements": [
    {
      "id": "e000000",
      "layer_id": "L00",
      "layer_order": 0,
      "stroke_width": 1.25,
      "style": {"tag": "ink", "tone": 0.42, "roughness": 0.18, "tint": [24, 31, 38]},
      "sampling": {
        "mode": "adaptive",
        "tolerance_px": 0.45,
        "min_subdivisions": 1,
        "max_subdivisions": 18,
        "width_scale": 1.0
      },
      "closed": true,
      "segments": [
        {"type": "cubic", "points": [[10.0, 20.0], [12.0, 22.0], [16.0, 18.0], [20.0, 20.0]]}
      ]
    }
  ]
}
```

段类型支持 `line`、`quadratic`、`cubic`，控制点数分别为 2、3、4。所有 JSON 写出使用排序键、紧凑分隔符和 UTF-8，因此同一状态总能得到逐字节相同的项目文件。

### 迁移表

- v1 → v2：补齐 `canvas`、`generator`、顶层 `sampling`、图层记录、`layer_order`、逐图元 `style` 与 `sampling` 固定默认值。
- v2 → v3：规范化 `layer_id`、图层顺序与元素顺序字段。
- 已是 v3 的项目再次迁移不做内容改动，规范化内容哈希不变。

## 二进制描边骨架格式

文件头固定 64 字节：

| 偏移 | 长度 | 字段 |
|---|---:|---|
| 0 | 8 | magic `NPRSKEL\0` |
| 8 | 2 | 字节序标记 `LE` / `BE`（作用于 payload 数值） |
| 10 | 1 | 索引宽度 `2` 或 `4` |
| 11 | 2 | 格式版本，大端 `u16` |
| 13 | 1 | 线宽编码 `1=f32`、`2=u16_fixed(1/256)` |
| 14 | 2 | flags / reserved |
| 16 | 4 | 总顶点数，大端 `u32` |
| 20 | 4 | 总索引数，大端 `u32` |
| 24 | 4 | 图层数，大端 `u32` |
| 28 | 4 | 图元数，大端 `u32` |
| 32 | 32 | payload SHA-256 |

payload 包含画布、种子、迁移后 schema、默认采样、生成器元数据、图层、逐图元控制点/段类型/风格/采样、由同一扁平化折线派生的骨架顶点，以及索引数组。

控制点使用 `f64`，默认线宽编码为 `f32`；`u16_fixed` 的声明线宽容差为 `1/512`。格式检查覆盖 magic、字节序、索引宽度、payload 哈希、计数一致性、索引越界、UTF-8 与截断。

## 扁平化与骨架

- 贝塞尔曲线使用确定性 de Casteljau 中点递归细分，以控制点到弦的最大距离作为停止误差估计。
- `min_subdivisions` 控制强制细分深度，`max_subdivisions` 控制安全上限。
- 容差越小，细分树只会被进一步切开，因此折线顶点数单调不减。
- 同一输入和参数不包含随机数或时间量，两次结果逐点相同。
- 骨架顶点直接引用扁平化折线顶点；索引由各段扁平折线边界构造，不独立求值曲线。
- 直线长度为零、控制多边形长度为零、控制点全部重合或曲线完全坍缩会返回 `E_OUTLINE_DEGENERATE`。

## 光栅化策略

预算与选择公式位于 `npr_edit/strategy.py`：

| 档位 | 单帧预算 | 内存预算 | 典型选择 |
|---|---:|---:|---|
| 200 | 12 ms | 64 MB | `per_element` |
| 2000 | 40 ms | 128 MB | `layer_merge` |
| 20000 | 220 ms | 512 MB | `tile_batch` |

策略账本包含每个候选的预估耗时、内存、误差、是否可行、得分、预算原因和最终指纹。选择结果按 `档位 + 场景统计 + 预算` 缓存；修改 2000 档预算只新增 2000 档缓存项，不会重选 200/20000。

实际渲染遍历也按选择执行：逐图元顺序、按图层排序或按 64 px 包围盒瓦片排序。逐图元精灵缓存和逐图层合成保证增量编辑仍能精确复用闭包外结果。

## 增量闭包与快照

- 编辑一个控制点、线宽或风格参数时，闭包固定为 `{该图元}` 与 `{该图元所属图层}`。
- 只重建该图元的几何与精灵，只重合成该图层；其他图元骨架哈希、渲染哈希与其他图层缓冲保持不变。
- 引擎会比较计划的闭包、几何实际重建集合、光栅实际重算集合和图层实际重算集合，不一致返回 `E_CLOSURE_MISMATCH`。
- 切换风格模式会重算光栅，但不会调用扁平化或骨架构建；几何哈希与骨架哈希必须保持不变。
- `RenderSnapshot` 为冻结对象，包含完整 BGRA、逐图层缓冲、逐图元精灵、扁平几何、全部哈希与版本戳。
- `SnapshotStore` 通过单次引用替换发布整份快照；读者只会看到旧快照或新快照。
- 读取指定版本或内容哈希不匹配时返回 `E_SNAPSHOT_STALE`。

## 持久化与恢复

```text
state/
  CURRENT                         # 原子改名的提交指针
  CLOSED                          # 只在正常关闭后出现
  generations/<generation>/
    project.json
    manifest.json
    snapshot.bgra
    geometry.json
    sprites.json
    sprites.bin
    layers/layer-0000.bgra
    COMMITTED                     # 已作为 CURRENT 发布过的代际标记
  generations/<uncommitted>/...    # 无 COMMITTED，恢复时移入孤儿
  tmp/...                          # 未完成临时写入
  orphans/RECOVERY.json
```

提交顺序：

1. 在 `tmp/` 写完整临时代际并 `fsync`。
2. 原子改名为 `generations/<generation>`。
3. 原子替换 `state/CURRENT`。
4. 写入该代际的 `COMMITTED` 标记。

因此：

- `CURRENT` 未更新前，所有新文件都不可见为稳定状态。
- 恢复只读取 `CURRENT` 指向的代际，并校验项目哈希、快照哈希、图像哈希、图层哈希、精灵像素哈希和完整清单。
- 无 `CURRENT` 但有历史代际时返回 `E_RECOVERY`，不会猜测某一代。
- 无 `COMMITTED` 的代际与 `tmp/` 内容移入 `orphans/`，并写入确定性的恢复报告。
- 强制终止由真实子进程 `os._exit(137)` 模拟，不使用伪退出标记。
- 正常关闭写 `closed:true` 清单并原子创建 `CLOSED`；再次恢复要求 `CURRENT`、代际与关闭标记三者哈希完全一致。
- 恢复后的 `NprEngine.from_snapshot` 保留恢复版本戳，下一次编辑必须显式携带该版本与快照哈希。

## 业务错误码

| 错误码 | 含义 |
|---|---|
| `E_SCHEMA_VERSION` | schema 过新、缺失或不可迁移 |
| `E_FORMAT_ENDIAN` | magic 或字节序标记错误 |
| `E_FORMAT_INDEX` | 索引宽度、计数、payload、截断等错误 |
| `E_FORMAT_WIDTH` | 不支持的线宽编码或越界 |
| `E_STROKE_SAMPLE` | 采样容差、细分或线宽配置非法 |
| `E_OUTLINE_DEGENERATE` | 零长、重合控制点或坍缩曲线 |
| `E_CLOSURE_MISMATCH` | 计划闭包与实际重算集合不一致 |
| `E_SNAPSHOT_STALE` | 版本或快照内容哈希已陈旧 |
| `E_GL_CONTEXT` | 本地隐藏 OpenGL 上下文/FBO 创建失败 |
| `E_EDIT_TARGET` | 元素、段、控制点或风格参数目标无效 |
| `E_RECOVERY` | 提交代际缺失、损坏或哈希不一致 |
| `E_WORKSPACE` | 工作区不可写或路径不安全 |

CLI 业务错误输出规范化 JSON 到 stderr；`E_GL_CONTEXT` 退出码为 `30`，`E_SNAPSHOT_STALE` 为 `31`，`E_CLOSURE_MISMATCH` 为 `32`。

## 确定性边界

以下内容不进入摘要或哈希：

- 实际墙钟耗时。
- 临时文件名、进程 ID、绝对工作区路径。
- 未提交写入的临时目录位置。

进入内容哈希的内容均使用显式排序、确定性浮点表示或规范化 JSON：

- 场景文件、每图元哈希清单。
- 二进制 payload 与文件头。
- 几何、骨架、逐图元光栅与逐图层内容哈希。
- 快照版本、模式、视口、全部图层/图元/骨架哈希。
- 持久化代际、提交指针、关闭标记和孤儿报告。

因此相同种子、参数和代码版本在支持平台上得到逐字节相同的摘要、场景、二进制、BGRA 与账本。


## 实际验证结果

验证日期：2026-11-20。当前执行环境为 WSL2/Linux，未安装本地 GLFW/OpenGL Python 包，因此完整业务链路使用软件 BGRA 后端运行；Windows 专属 WGL 启动代码已实现，但本机未执行真实 Windows 显卡路径验证。

已执行：

```text
python -m unittest discover -s tests -v
```

结果：`19` 个测试全部通过。

已执行：

```text
python -m npr_edit demo --workspace /tmp/npr-demo-run --seed 20261120 --viewport 256x192
```

结果：

- 三档生成均二次逐字节一致。
- 三档 JSON ↔ 二进制往返均在声明容差内，`roundtrip_exact_scene=true`。
- v1 迁移二次哈希一致。
- 三档策略分别选中 `per_element`、`layer_merge`、`tile_batch`。
- 三档增量编辑闭包均与实际重算集合完全一致。
- 200 档切换 `sketch`、`cartoon` 时几何/骨架哈希不变。
- 真实子进程退出码为 `137`，未提交目录被回收为孤儿。
- 恢复版本与基线哈希一致，恢复后的编辑使用恢复版本戳。
- 最终 `CLOSED` 标记通过验证。
- 全部错误码探针均命中。
- OpenGL 探测返回 `E_GL_CONTEXT`，退出码 `30`，并记录软件降级。

本次摘要内容哈希：

```text
84b3375d70555747d7160776199044b1a671e26163010599155d9a47ea6d2e32
```

已执行：

```text
python -m npr_edit verify --workspace /tmp/npr-two-run --seed 20261120 --viewport 256x192
```

结果：

```json
{
  "summary_byte_identical": true,
  "summary_hash_a": "bce4a71e320fe41002b04bb1428d6c3e0928664ef731746ed5e21744d113f31d",
  "summary_hash_b": "bce4a71e320fe41002b04bb1428d6c3e0928664ef731746ed5e21744d113f31d",
  "deterministic_file_count": 55,
  "tree_files_identical": true,
  "differing_files": [],
  "verified": true
}
```

20000 档演示执行的是全量 `outline` 渲染加单图元增量编辑；三种模式的共享几何验证在 200 档完整执行。这样固定序列仍覆盖三档压力与闭包语义，同时把本地纯软件渲染时间控制在可重复演示的范围内。
