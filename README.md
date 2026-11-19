# CAD Surface Lab

本地确定性 NURBS 曲面编辑、显示、版本对比与崩溃恢复示例。项目不访问网络、不依赖远程服务或外部模型文件；输入模型由固定种子生成。几何内核、软件光栅显示、测试和验收入口均为 Python 3.11+。

## 快速开始（Windows）

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python run.py demo
```

也可以执行 `run_demo.cmd`。单条验收入口为 `python run.py demo`，它会依次完成：

1. 用固定种子生成两次模型并比较逐字节输出。
2. 生成 3 个 NURBS 曲面片，输出模型哈希。
3. 提交控制点编辑，验证局部失效闭包与局部重算集合。
4. 执行撤销、重做、撤销后新分支和重做截断。
5. 从操作日志逐次重放单个曲面片，与当时直接产出的几何/法线哈希比较。
6. 用独立子进程写入未提交 staging 缓存后执行 `os._exit(23)`，模拟强制终止。
7. 重启恢复最近一次已提交状态，回收并记录孤儿缓存。
8. 在恢复后的工作区继续编辑，并与未中断参考工作区逐项比较。
9. 验证正常退出关闭标记，并覆盖业务错误码链路。
10. 写出 `artifacts/cad_demo/report.json` 和三种显示模式 PPM 截图。

常用命令：

```powershell
python run.py generate --seed 20261119
python run.py create artifacts/manual_workspace --grid 16
python run.py edit --root artifacts/manual_workspace --patch p00 --u 4 --v 4 --value -2.2 -0.5 0.45
python run.py undo --root artifacts/manual_workspace
python run.py redo --root artifacts/manual_workspace
python run.py compare --root artifacts/manual_workspace --first 0 --second 1
python run.py export --root artifacts/manual_workspace --version 1 --mode shaded_surface --output artifacts/view.ppm
python run.py close --root artifacts/manual_workspace
```

## 架构

主要代码位于 `src/cad_surface/`：

| 模块 | 职责 |
| --- | --- |
| `model.py` | 固定种子确定性样本、相机/材质通道、序列化与内容哈希 |
| `nurbs.py` | NURBS 数据模型、节点/阶数校验、统一求值过程 |
| `geometry.py` | 网格/等参线/法线缓存、控制点依赖映射、精确失效闭包 |
| `state.py` | 带版本状态的聚合哈希与显示缓冲派生 |
| `history.py` | 操作日志、撤销/重做栈、分支截断、父版本链 |
| `render.py` | 确定性软件光栅器，三种显示模式与显示缓冲哈希 |
| `persistence.py` | 原子提交目录、清单版本戳、缓存哈希、恢复与孤儿回收 |
| `engine.py` | 应用门面、编辑、恢复、历史补片重放、版本比较 |
| `cli.py` | 单命令演示、交互命令和错误报告 |

### 统一求值

每个采样点都由同一个 `NURBSPatch.evaluate(u, v)` 得到曲面点、`du`、`dv`、法线和节点区间元数据。`evaluate_grid()` 只循环调用该函数；三角网格顶点、等参线采样点直接引用同一采样网格，不另写求值实现。缓存记录阶数、全部节点区间、采样顺序和求值口径。同参数重复求值的规范化 JSON 字节完全相同。

### 局部失效闭包

编辑控制点 `(i, j)` 时，先用求值得到的非零基函数支撑（同时覆盖法线所需的一阶导数支撑）构造 `cp -> vertex` 依赖。闭包包含：受影响曲面片、受影响采样顶点、与顶点关联的等参线采样点、任一顶点受影响的三角面。

只对这些顶点调用统一求值过程；闭包外点、导数、法线逐字节复制。`recompute_trace` 同时记录声明闭包与实际重算集合，不一致立即返回 `E_CLOSURE_MISMATCH`。未改动的曲面片仅建立新版本缓存键，不重新求值。

### 版本与历史重放

每次提交编辑创建全局递增版本，操作记录包含父版本、曲面片键、控制点索引、前坐标、后坐标和结果状态哈希。`reconstruct_patch_at_version()` 从版本 0 的单补片缓存沿父链只重放目标补片的操作，每次都只计算该控制点的局部闭包；随后与目标版本直接缓存比较几何、法线和逐点偏差，不会重建整个模型。

### 持久化布局

```text
workspace/
  MANIFEST.json                 # revision + current state hash + undo/redo + closed
  RECOVERY.json                 # 最近一次打开时的恢复结论
  SESSION_REPORT.json           # 正常退出报告（仅在受控关闭时写）
  commits/
    v000000/
      commit.json               # 状态/操作/缓存文件哈希与提交内容哈希
      state.json
      operation.json            # v0 无操作
      patches/p00.json ...      # 版本 + 曲面片键索引的完整缓存
  staging/                      # 未发布提交，启动时作为孤儿回收
```

提交顺序为：在 staging 写入并 fsync 全部状态、操作和缓存文件；原子重命名成 `commits/vNNNNNN`；最后原子替换 `MANIFEST.json`。只有清单指向的提交被视为稳定状态。清单同时校验 `revision` 与内容哈希，旧会话不能覆盖新快照，否则返回 `E_SNAPSHOT_STALE`。每个缓存文件同时校验文件哈希、缓存内容哈希、版本/补片键和几何哈希，任一不符返回 `E_CACHE_HASH`。

### 受控退出与强制终止

- `close()` 写完整清单（`closed: true`）和 `SESSION_REPORT.json`。
- `os._exit()` 或进程被杀时不会写关闭标记；重启只加载清单中最后已提交版本。
- `staging/` 中的未提交缓存被删除并写入 `RECOVERY.json` 的 `orphan_paths`。
- 若重启时连清单也缺失但仍有提交目录，返回 `E_SNAPSHOT_STALE`，不会猜测或伪造恢复/退出报告。

### 软件显示

`render.py` 使用固定相机的确定性软件光栅器，输出 RGB 与深度缓冲，并提供 `control_net`、`shaded_surface`、`isoparametric` 三种模式。显示缓冲哈希同时覆盖颜色与深度。项目声明 PyOpenGL 依赖，但验收路径采用软件光栅，避免 Windows CI/无窗口环境下离屏上下文差异。

## 错误码

| 错误码 | 触发示例 |
| --- | --- |
| `E_SCHEMA_VERSION` | 模型/状态 schema 不匹配、阶数/控制点数不匹配、节点长度不匹配 |
| `E_KNOT_MONOTONIC` | 节点向量出现下降或空参数域 |
| `E_GEOM_STALE` | 参数越界、缓存与当前控制点/版本不一致、空重做等无效几何操作 |
| `E_CLOSURE_MISMATCH` | 闭包外几何变化或实际重算集合与声明闭包不一致 |
| `E_UNDO_EMPTY` | 撤销空栈（重做空栈也使用该业务栈错误） |
| `E_SNAPSHOT_STALE` | 清单内容哈希/revision/状态哈希过期、强制终止后清单缺失 |
| `E_CACHE_HASH` | 补片缓存文件、缓存封套或内部几何哈希不匹配 |

所有校验发生在修改已提交状态之前；失败不会改写已有清单或提交目录。

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试覆盖：确定性生成与序列化、节点/阶数校验、统一求值重复性、局部闭包逐字节不变性、实际重算集合、单补片历史重放、撤销/重做/分支截断/栈深度、正常关闭恢复、强制终止恢复、缓存篡改、清单过期/缺失、revision 竞争和全部业务错误码。

## 实测记录

验证日期：2026-11-19。环境：Python 3.11.0rc1、NumPy 1.26.4、PyOpenGL 3.1.7；代码避免 POSIX 专属路径/替换逻辑。实际运行：

```text
python run.py demo
Result: PASS
Tests: 9 passed
Model hash: 6b9135986b23648c2b892d3feebc1f8bafb599db936d7a61c13fece04cf7341e
Initial geometry hash: e9dc27f01aaced8b81542075f787db4095d2d11a90e8955087e0e2d26cc464c0
Closure: 100 / 256 vertices, 200 triangles, actual set == declared set
Historical patch replay: geometry/normal hashes equal, max deviation 0.0
Forced termination: worker exit 23, one staging orphan reclaimed
Recovered version: 3, state hash ef11e0ad78f44e8031ed896e7d8b03f6de0bbf17c26c67b40f9b1c0ffde6b1c1
Recovery deviation: point 0.0, normal 0.0, all three display buffers identical
Post-recovery continuation vs uninterrupted: state/geometry/normal/display hashes equal
Controlled close: closed flag and CONTROLLED_EXIT recognized
```

详细机器可读结果见 `artifacts/cad_demo/report.json`，显示证据见 `artifacts/cad_demo/screenshots/`。
