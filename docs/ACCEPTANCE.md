# 本地验收记录

## 执行命令

```bash
python3 -m venv .venv --without-pip
curl -fsSL https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_acceptance.py --workspace .
```

实测环境：Linux/WSL、Python 3.11.0rc1、NumPy 2.4.6、pytest 9.2.0。目标发布环境为 Windows 10/11 + Python 3.11/3.12；Windows 安装入口为 `scripts/setup_windows.ps1`。未使用 Docker 或容器。

机器可读完整报告：`artifacts/acceptance/acceptance_report.json`。
错误码、哈希分项报告：`artifacts/acceptance/error_report.json`、`artifacts/acceptance/hash_report.json`。

本次报告稳定哈希：

```text
report_hash = a91ba94dab77f99b850e2e72aa63b6a435408e9f01e261764a5412796a7f9c73
```

连续两次运行报告哈希一致。pytest 结果：`50 passed`。

## 验收总表

| 检查 | 结果 |
| --- | --- |
| 四种 endian/index 组合 + v1 旧线宽往返 | PASS |
| v1/v2 迁移、线稿等价与幂等 | PASS |
| 三模式共享几何/骨架 | PASS |
| 精确编辑、局部重算、撤销复原 | PASS |
| 快照原子提交、真实强制终止恢复、缓存复用 | PASS |
| 版本对比归因、持久化与复算 | PASS |
| 7 个业务错误码注入 | PASS |

## 格式往返

项目内容哈希：

```text
json://sha256:49b44d019e1cdf30366de4147148f8c6da4c1f77df6b5743e1b4b78f8ef31cb1
geometry: 0c62c7f06ec5dc442c31935a930266099f1f0037a2a4658e8c3b8a84510d5da5
```

| 版本 | 字节序 | 索引位宽 | 二进制哈希 | 最大顶点误差 |
| --- | --- | ---: | --- | ---: |
| v2 | little | 16 | `0b2accb32d59ef4f2ca959d5ef68190e1b346ba719d10659b41d79796ff84e6f` | 0.0 |
| v2 | little | 32 | `b43db001bc472a061aa39bcb2d350211351ae9d5b1238a1021bd6baf80670d34` | 0.0 |
| v2 | big | 16 | `deecea9c044ddbbe464324d44aa1a1a4f7e8c1ccd63478b4ed704c580e54b96c` | 0.0 |
| v2 | big | 32 | `8ec3bcd707ef80898c27fb8e003f6adc6bc07f4689ad9d64ea0ae04a2b5cac30` | 0.0 |
| v1 | big | 16 | `abf103a11be801d4ba41f2c2b9238d059e2077dafe0e3cbc4487f5d544216046` | 0.0 |

所有带 metadata 往返的 JSON 哈希均回到 `49b44d...31cb1`。

## 渲染与编辑

| 模式 | 像素 SHA-256 | PNG SHA-256 | 切换后扁平化构建 | 骨架构建 |
| --- | --- | --- | ---: | ---: |
| lineart | `95e192754dde80da1c18164820aecd1b8c17c7a41aea43872c825c68e384656e` | `089c655fb6c7814620439245f52e5e73b6882d94e8c9e3151fc754b3d11c81bd` | 7（首次） | 7（首次） |
| sketch | `697096195e65d092261368b9d26d3ce98129d32f95e0aa0ca2130f1c99542372` | `e389968a936ae36d76f01684b5ed2d1171be2f35b79d696206f4f3dc7070582b` | 0 | 0 |
| toon | `0bd1172dce335be51ab5af169fe88ab1f86079610a6548a03adde2017bc76b5b` | `c57a9f63e8e5406a94f239c76297e3b99ee977217dae90cc02f330978d68ef2f` | 0 | 0 |

控制点编辑 `tree_crown` 后：

- 几何哈希变化集合：`["tree_crown"]`
- 渲染哈希变化集合：`["tree_crown"]`
- 其他图元顶点逐点完全相等
- 撤销后项目、几何、逐图元渲染和整图像素哈希全部恢复

## 快照与强制终止

首个提交快照：

```text
stamp  = 000000000001
hash   = 3f341ddb8b172c3aa4887010ec6a20891e9f2b8e37ff4f68ace26552a77c8a53
```

子进程在 staging payload 写完后执行 `os._exit(23)`。重启恢复结果：

- 子进程返回码：`23`
- 可见已提交快照数：`1`
- 最近提交仍是首个快照
- staging 孤儿被回收
- 未发布的新版本未被伪装成成功提交

随后正常提交 stamp `000000000002`，恢复原始 RGBA 缓存后整图重建构建次数为 `0`，像素哈希与提交前一致。

## 版本对比

对比 v1 迁移视图与编辑后的 v2 版本：

```text
difference kinds = control_point, format_version
sources          = edit, migration
suspicious       = []
comparison_hash  = 1e2319580a4dc714fea4a2e551b01cbe8850fd457669798af9e2c5794cc21730
recomputed_equal = true
```

并排图：`artifacts/acceptance/comparison_side_by_side.png`。

## 错误码注入

| 错误码 | 注入方式 | 实际结果 |
| --- | --- | --- |
| `E_SCHEMA_VERSION` | schema=99 | PASS |
| `E_FORMAT_ENDIAN` | marker=`deadbeef` | PASS |
| `E_FORMAT_INDEX` | 最后索引=65535 | PASS |
| `E_STROKE_SAMPLE` | `curve_steps=0` | PASS |
| `E_UNDO_EMPTY` | 新编辑器直接撤销 | PASS |
| `E_SNAPSHOT_STALE` | 旧版本戳提交 | PASS |
| `E_RENDER_MISMATCH` | 注入不一致图元渲染哈希 | PASS |

## 产物索引

- `samples/current_v3.json`、`legacy_v1.json`、`legacy_v2.json`
- `samples/skeleton_v2_{l16,l32,b16,b32}.npbs`
- `samples/skeleton_v1_{l4,b2}.npbs`
- `samples/invalid_*.npbs`、`invalid_stroke_sample.json`
- `artifacts/acceptance/preview_{lineart,sketch,toon}.png`
- `artifacts/acceptance/undo_restored.png`
- `artifacts/acceptance/comparison_side_by_side.png`
- `artifacts/acceptance/acceptance_report.json`
- `artifacts/acceptance/error_report.json`
- `artifacts/acceptance/hash_report.json`
