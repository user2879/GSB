# PickWorkbench：确定性拾取与空间操作验证工作台

本项目在本地合成三维样本上验证射线拾取、屏幕空间点选、拖拽平面投影、栅格吸附、异常注入和首个失败定位。实现只依赖 Python 3.11/3.12 与 NumPy，不使用 GPU、Docker、数据库、远程服务或真实素材。

## 快速开始

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m pickworkbench demo --out artifacts\demo
```

Linux/macOS 的等价命令为 `python3 -m venv .venv`、`source .venv/bin/activate` 和相同的 `python` 命令。也可直接运行 `pip install -e .` 后使用 `pickworkbench demo --out artifacts/demo`。

`demo` 是唯一的端到端验收入口。它生成正常、退化射线、零长度法线、重复对象 ID、越界优先级、非等距吸附间距和非法注入配置七个固定样本，执行 22 条查询，输出逐查询哈希、结构化报告和最小反例。命令内部先生成两遍并逐字节比较，只有完全一致才写出结果。

## 命令

```text
python -m pickworkbench generate --out artifacts/sample
python -m pickworkbench validate --scene scene.json --queries queries.json \
    --injection injection.json --out report.json [--expect E_RAY_DEGENERATE]
python -m pickworkbench witness --witness witness.json
python -m pickworkbench demo --out artifacts/demo
```

- `generate`：按固定种子生成 `scene.json` 与 `queries.json`，打印对象数、联合包围盒范围和查询条数。
- `validate`：应用注入配置后运行整批查询；单条查询失败会形成隔离的错误记录，不会终止后续查询。
- `witness`：单独重跑最小反例中的有效注入后场景和查询。
- `demo`：执行固定样本集、全链路操作和重复性检查。

## 演示输出

```text
artifacts/demo/
  manifest.json
  report.json                     # 顶层索引、总哈希、各查询哈希
  cases/00_normal/
    scene.json                    # 注入前基准场景
    queries.json
    injection.json
    records.json                  # 逐查询记录与内容哈希
    report.json                   # 首个失败、最小反例、汇总
  cases/01_degenerate_ray/ ...    # 其余场景同构
```

当前固定配置为 `seed=20261119`、12 个对象（球体/AABB/凸多面体各 4 个）、10 条射线、8 条点选、4 条拖拽，视口 `1280x720`，默认点选容差 `5 px`。

本工作区最后一次实际验证：

| 项目 | 结果 |
|---|---|
| 测试 | `34` 项全部通过 |
| 正常场景 | `22/22` 查询通过 |
| 退化射线 | 首个受影响 `q-000-ray`，`E_RAY_DEGENERATE`，其余 `21` 条继续执行 |
| 零长度法线 | 首个受影响 `q-001-ray`，`E_NORMAL_ZERO` |
| 重复对象 ID | 首个受影响 `q-000-ray`，`E_OBJECT_DUP_ID`，最小反例含两个同 ID 对象 |
| 越界优先级 | 首个受影响 `q-000-ray`，`E_PRIORITY_RANGE` |
| 非等距吸附间距 | 首个受影响 `q-018-drag`，`E_SNAP_SPACING` |
| 非法注入配置 | `E_INJECT_CONFIG`，批处理被明确拒绝 |
| 输出重复性 | 两次运行所有 `42` 个文件逐字节一致 |
| `scene_hash` | `dd1b90668976a4b26b21a589b7477c8ee92810925b7d2f4234b56b7fa429de86` |
| `queries_hash` | `d4f1235830e92f44d056a780066531c8205242f5caf6331931fdd2de24156dcd` |
| `overall_hash` | `234304467d21e591385233a94839ca8831ab07ce5c1106ad4a03ba304b6dbfb0` |

## 核心语义

### 射线求交

射线方向在求交前归一化为单位向量，`hit_distance` 是世界空间距离。三类对象分别使用：

- 球体：局部坐标二次方程，`sphere_quadratic`。
- 轴对齐盒：世界 AABB slab 测试，`aabb_slab`。
- 凸多面体：世界半空间裁剪，`convex_halfspace_clip`。

最终命中对象按 `pick_priority` 降序、`hit_distance` 升序、`object_id` 字典序升序选择。该规则显式写入每条记录，确保优先级相同时也有确定次序。

### 屏幕空间点选

每个对象投影为圆或屏幕 AABB，点选容差扩展该屏幕形状得到候选集。候选对象仍使用点选像素对应的等价世界射线做精确求交，最终选择规则与射线路径相同。任何真实射线命中点都会落在对象投影轮廓内，因此声明容差下候选集包含等价射线路径的全部命中对象，两条路径的对象 ID、距离和命中点一致。`tolerance_probe` 使用 `tolerance+8 px` 复算候选集和赢家，记录候选集是否变化以及赢家是否不变。

### 拖拽与吸附

起点和终点屏幕坐标分别转换为相机射线，与给定拖拽平面求交，得到世界位移并加到给定起点。吸附开启时，仅对配置的对齐轴按单一正间距做绝对栅格取整；位置的其他分量、朝向及未吸附位置均显式保留。关闭吸附的重放结果必须等于记录中的 `unsnapped_position`。

### 错误码

| 错误码 | 含义 |
|---|---|
| `E_RAY_DEGENERATE` | 射线方向为零向量 |
| `E_NORMAL_ZERO` | 凸多面体平面或拖拽平面法线长度为零 |
| `E_OBJECT_DUP_ID` | 两个及以上可拾取对象使用相同 ID |
| `E_PRIORITY_RANGE` | 拾取优先级超出场景声明范围 |
| `E_SNAP_SPACING` | 吸附间距非正、非数值或非等距 |
| `E_PICK_MISMATCH` | 点选候选路径与等价射线路径结果不一致 |
| `E_INJECT_CONFIG` | 注入配置或配置引用无效 |

失败查询记录 `status=error`，携带错误码、明确消息和输入片段，不包含赢家，也不会进入 `passed` 计数。`first_affected` 按查询序号升序查找；同一查询内按对象 ID、对象索引及固定异常码次序查找。

### 最小反例

每个失败场景输出 `witness.json`，包含源输入哈希、有效注入后场景切片、相关查询、首次复现结果，以及逐项删除检查。删除反例中的任一对象或查询后，同一错误不再出现，故 `minimal=true`。可用 `python -m pickworkbench witness --witness ...` 独立复核。

### 哈希与可复现性

JSON 统一按 UTF-8、键排序、紧凑规范形式送入 SHA-256；浮点输出固定到 12 位小数并将负零归一为零。每个查询记录对除 `content_hash` 外的全部字段求哈希，批记录再对完整记录数组求哈希。报告不含时间戳、绝对路径或随机状态，因此同一输入和注入配置产生逐字节相同的文件与哈希。

更详细的算法与不变量见 [docs/design.md](docs/design.md)。

## 测试

```text
python -m unittest discover -s tests -v
```

测试覆盖：生成器逐字节重复、三类求交、非单位射线归一化、优先级/距离/ID 决胜、点选与等价射线一致、容差只改变候选集、拖拽确定性、吸附复原、朝向不变、全部七类错误码、首个失败定位、单查询隔离、最小反例复现、删除最小性、非法注入拒绝，以及 `demo` 两次运行的 stdout、全部文件与哈希逐字节一致。
