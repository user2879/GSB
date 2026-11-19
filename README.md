# Level Workbench

本地桌面关卡编辑工作台，用于验证屏幕拾取、跨版本拾取记录归一、历史记录复现、空间操作、原子写出和分块中断恢复。项目只使用 .NET 8 BCL 与 Windows 自带 WinForms，不依赖图形库、外部服务、端口或数据集。

## 交付内容

- `src/LevelWorkbench.Core`：场景/记录解析、摘要绑定、相机射线、三角形求交、过滤、框选/拖动/撤销、重放、迁移、原子写入和有界并行调度。
- `src/LevelWorkbench.SampleGen`：确定性场景、多版本记录、非法样本和拾取脚本生成器。
- `src/LevelWorkbench.Cli`：无界面验收入口，所有错误输出 `E_xxx` JSON。
- `src/LevelWorkbench.Desktop`：Windows WinForms 工作台，支持视口拾取、框选、拖动、缩放/平移、撤销、记录写入、历史重放和迁移。
- `tests/LevelWorkbench.SelfTest`：无第三方测试框架的 39 项自动验证。
- `sample-data/`：已由生成器生成，可直接用于验收；重新生成会得到逐字节相同的结果。

## Windows 运行

要求 Windows 10/11 与 .NET 8 SDK：

```powershell
dotnet --info
.\scripts\generate-samples.ps1
dotnet build .\LevelWorkbench.sln
dotnet run --project .\src\LevelWorkbench.Desktop -- `
  --scene .\sample-data\scene.json `
  --records .\sample-data\records\mixed.json `
  --script .\sample-data\pick-script.json `
  --state .\workbench-state `
  --out .\workbench-output\picked-records.json
```

桌面操作：

- 左键拾取；`Ctrl` 加选/减选。
- `Shift` + 左键拖动框选。
- 在命中代理上左键拖动，按相机朝向拾取平面更新位置并持续重算命中。
- 中键平移，滚轮围绕鼠标位置缩放。
- `Ctrl+Z` 撤销位置和选中集，`Esc` 取消拖动或清空选择。
- 右侧按钮可加载场景/记录、运行拾取脚本、重放历史、迁移记录。每次最终命中原子追加到 `--out` 指定文件，脚本结果原子写到 `workbench-output/script-picked-records.json`。

Windows 图形界面需要在 Windows 会话中运行。本仓库当前验证环境为 WSL/Linux，因此桌面项目完成了 `net8.0-windows` 交叉编译验证，未执行 Windows 图形会话下的鼠标人工点击。

## 生成与验证

Linux/WSL：

```bash
./scripts/generate-samples.sh
dotnet build LevelWorkbench.sln
dotnet run --project tests/LevelWorkbench.SelfTest
```

Windows：

```powershell
.\scripts\generate-samples.ps1
dotnet build .\LevelWorkbench.sln
dotnet run --project .\tests\LevelWorkbench.SelfTest
.\scripts\verify.ps1
```

脚本生成的操作没有任何时间戳或随机数：场景几何、剖分索引、快照摘要、记录命中点、脚本步骤和非法样本均可逐字节复现。

## CLI 验收命令

```bash
# 校验并展示归一化结果
dotnet run --project src/LevelWorkbench.Cli -- validate \
  --scene sample-data/scene.json --records sample-data/records/mixed.json

# 分块重放，成功记录也写入报告；失败返回 E_REPLAY
dotnet run --project src/LevelWorkbench.Cli -- replay \
  --scene sample-data/scene.json --records sample-data/records/mixed.json \
  --state workbench-state --report workbench-output/replay-report.json

# 执行拾取/选择/框选/拖动脚本，块状态可恢复，最终记录原子写出
dotnet run --project src/LevelWorkbench.Cli -- pick \
  --scene sample-data/scene.json --script sample-data/pick-script.json \
  --state workbench-state --out workbench-output/picked-records.json

# 读取 v1/v2/v3 并升级为当前 v3；已存在旧版本文件时先归档
dotnet run --project src/LevelWorkbench.Cli -- migrate \
  --scene sample-data/scene.json --records sample-data/records/legacy-v1.json \
  --out workbench-output/records-v3.json
```

业务退出码：`0` 成功，`2` 业务错误（JSON 中含 `E_xxx`），`130` 在块边界处理终止信号并写出 `interrupt.json`。状态版本标识由场景几何规范化摘要与记录/脚本集合规范化摘要共同构成。

## 实际验证结果（2026-11-19，WSL Ubuntu 22.04，.NET SDK 8.0.428）

- `dotnet build LevelWorkbench.sln`：成功，0 warning / 0 error；同时交叉编译 `LevelWorkbench.Desktop` 的 `net8.0-windows` 目标。
- `dotnet run --project tests/LevelWorkbench.SelfTest`：`PASS 39 checks`，包含确定性生成、非法输入、E_IDENT 冲突候选、串并行逐字段一致、过滤、缩放/平移、法线、框选、拖动撤销、当前位姿快照重放、拾取/重放块复用与中断恢复、块摘要篡改、`SIGTERM` 退出码 130、迁移归档、未知字段保留、原子写失败保留旧文件。
- 混合版本重放：4/4 条成功（v1 + v2 + 2×v3），2 个块完成。
- 拾取脚本：6 个步骤，产出 5 条当前版本记录；随后离线重放 5/5 成功。
- 非法单位记录：CLI 返回 `E_UNIT` JSON，退出码 2，未写出记录文件。
- 确定性样本树整体摘要：在 `sample-data/` 内执行 `find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum` 得到 `e7b5b82e2d81a09cf9864bd232e2a2c70d41a975e23c20336af850b93b2edfc6`；重新生成可复现。该值只是样本快照校验，不是代码 API。

详细格式与设计见 [docs/SCHEMA.md](docs/SCHEMA.md) 和 [docs/TECHNICAL.md](docs/TECHNICAL.md)。

## 关键语义

- 旧版 v1：`proxyNumber + faceIndex + hitPointCm + snapshotDigest`，摘要只匹配当前几何；摘要无匹配、重复匹配或与代理编号冲突都返回 `E_IDENT`，不会按编号猜测。
- v2：使用 `stableId` 绑定对象，记录米制命中点、相机侧法线、屏幕射线距离和图层/过滤字段；可选 `faceIndex`，缺失时由命中点和法线唯一反推。
- v3：在 v2 基础上保存视口尺寸、缩放和平移，重放时使用记录中的视口参数。
- `screenDistanceMeters` 定义为相机沿单位射线的世界距离，单位米，与求交参数 `t` 相同。
- 面法线统一归一化，并翻转到与射线方向相反的一侧，即朝向相机。
