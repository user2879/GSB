# NPR 线稿与风格化渲染编辑应用

这是一个可在 Windows 10/11 上运行的确定性 NPR 编辑应用核心与命令行程序。它覆盖项目 JSON、二进制描边骨架、schema 迁移、三种风格化渲染、精确增量编辑与撤销、崩溃安全快照、版本对比以及业务错误码。

当前交付采用 headless 应用形态：所有编辑和渲染能力由稳定 Python API 提供，CLI 可直接执行验收操作；没有引入 GUI、Docker、网络素材或 Pillow 等额外图像依赖。

## 快速开始

### Windows PowerShell

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
```

也可以手工执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\generate_samples.py --out samples
.\.venv\Scripts\python.exe scripts\run_acceptance.py --workspace .
```

Python 3.11 也可使用。若 `py -3.12` 不存在，安装脚本会自动尝试 3.11。运行期只依赖 NumPy；pytest 仅用于开发测试。

### 当前验证环境

本次实测在 Linux/WSL 上执行，代码路径和原子操作均使用跨平台标准库，并提供了 Windows PowerShell 入口。实际结果见 `artifacts/acceptance/acceptance_report.json`。

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_acceptance.py --workspace .
```

当前结果：**50 项 pytest 全过；7 组验收检查全过；7 个业务错误码全部通过注入验证。**

## 功能入口

| 能力 | API | CLI |
| --- | --- | --- |
| 生成确定性样本 | `npr_editor.samples.write_samples` | `python -m npr_editor samples --out samples` |
| JSON/骨架往返 | `encode_binary` / `decode_binary` | `python -m npr_editor roundtrip ...` |
| v1/v2 → v3 迁移 | `migrate_document` | `python -m npr_editor migrate ...` |
| 三模式渲染 | `RenderEngine.render` | `python -m npr_editor render ...` |
| 编辑与撤销 | `Editor` | Python API |
| 原子快照 | `SnapshotStore` | `python -m npr_editor snapshot/restore ...` |
| 版本对比 | `compare_versions` / `ComparisonStore` | `python -m npr_editor compare ...` |
| 完整验收 | `run_acceptance` | `python -m npr_editor acceptance` |

示例：

```bash
# JSON -> v2 / big-endian / 16-bit 索引 -> JSON
.venv/bin/python -m npr_editor roundtrip \
  --project samples/current_v3.json \
  --out artifacts/roundtrip_be16 \
  --endian big --index-width 2 --format-version 2

# 三种模式
.venv/bin/python -m npr_editor render \
  --project samples/current_v3.json --mode toon --out artifacts/toon.png

# 快照与恢复
.venv/bin/python -m npr_editor snapshot \
  --project samples/current_v3.json --workspace artifacts/demo_ws --mode lineart
.venv/bin/python -m npr_editor restore \
  --workspace artifacts/demo_ws --out artifacts/restored.png --mode lineart
```

## 交付内容

- `npr_editor/model.py`：schema v3 项目模型、float32 控制点规范化、JSON 内容哈希。
- `npr_editor/migration.py`：固定迁移表与幂等性。
- `npr_editor/geometry.py`：共享曲线扁平化、描边骨架、签名缓存。
- `npr_editor/binary_format.py`：大小端、16/32 位索引、v1/v2 线宽编码和严格解析。
- `npr_editor/render.py`：纯轮廓、素描、卡通三种共享几何的确定性光栅器。
- `npr_editor/editing.py`：控制点/线宽/风格/采样编辑与撤销。
- `npr_editor/snapshots.py`：staging、COMMITTED、目录原子发布、孤儿回收和缓存校验。
- `npr_editor/compare.py`：逐图元差异、来源归因、可疑项、并排图和可复算持久化。
- `npr_editor/errors.py`：稳定业务错误码。
- `samples/`：多图元、多图层、v1/v2/v3、四种 endian/index 组合及损坏样本。
- `docs/FORMAT.md`：格式、协议、错误码与兼容规则。
- `docs/ACCEPTANCE.md`：本次实测哈希、错误码和强制终止结果。

## 设计要点

- 所有控制点进入内存时量化为 `float32`；格式往返逐点误差实测为 `0.0`。
- JSON 哈希使用排序键、紧凑分隔符的 canonical JSON；二进制哈希覆盖完整字节。
- 几何缓存键包含图元控制点与采样配置；模式切换只读取同一 `StrokeSkeleton`，不会触发扁平化或骨架重建。
- 渲染缓存键包含几何哈希、当前模式相关风格指纹、有效线宽和图元标签；因此局部线宽变化只改变该图元的渲染哈希。
- 快照只有在 staging 中写完全部文件、manifest 和 `COMMITTED` 后才原子改名为正式目录；无 `COMMITTED` 的目录一律视为孤儿。
- 快照恢复会校验项目内容哈希、几何哈希、缓存键和原始 RGBA 哈希，不盲目复用旧缓存。
- 对比来源仅来自可审计 trace：编辑、迁移、格式转换；其余几何差异进入 `suspicious_items`。

详细规范见 `docs/FORMAT.md`，实测记录见 `docs/ACCEPTANCE.md`。
