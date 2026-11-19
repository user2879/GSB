# Windows 本地环境运行说明

## 1. 环境准备（一次性）

1. 安装 Python 3.11 或更高版本，安装时勾选 **Add python.exe to PATH**，
   并保留默认的 **tcl/tk and IDLE**（提供 tkinter）。
2. 打开 PowerShell 或 CMD，确认：

   ```bat
   python --version
   pip --version
   ```

   若系统里 `python` 指向其它发行版，可用 `py -3.11` 替代 `python`。
3. 安装唯一第三方依赖：

   ```bat
   pip install numpy
   ```

   tkinter 与 sqlite3 均随标准库提供，无需额外安装。

## 2. 运行测试与一键验收

```bat
cd path\to\uidfc5909d3-default-300
python -m unittest discover -s tests -v
python -m bench.acceptance --workdir run\acceptance --json-out artifacts\acceptance_summary.json
```

## 3. 常用命令（全部输出为单行确定性 JSON）

```bat
set MW=python -m bench.mw.cli --workdir run
%MW% init     --spec samples\basic\spec.json --events samples\basic\events.ndjson
%MW% ingest   --spec samples\basic\spec.json --events samples\basic\events.ndjson --drain
%MW% query    --spec samples\basic\spec.json
%MW% recompute --spec samples\basic\spec.json
%MW% latency  --spec samples\basic\spec.json
%MW% snapshot --spec samples\basic\spec.json
%MW% set-params --spec samples\basic\spec.json --new-spec path\new_spec.json
%MW% inject   --spec samples\resume\spec.json --events samples\resume\events_part2.ndjson --kind regression --partition p0 --seq 2 --ts-ms 10
%MW% gui
```

PowerShell 中请改用 `$env:` 变量或直接写完整命令，不要使用 CMD 的 `set` 语法。

## 4. 运行时目录

`--workdir` 下自动创建三类目录，均可通过 CLI 参数单独指定：

- `ledgers/ledger.db`：SQLite 权威账本（事件、缓存状态、查询账本、水位/迟到日志）。
- `snapshots/snapshot.json`：带整数格式版本与 SHA-256 内容摘要的会话快照。
- `logs/events.ndjson`：结构化日志（每行一个规范 JSON，含错误码）。

## 5. 无图形界面环境

在没有 tkinter 的裁剪环境（如部分 Linux 容器）中，`gui` 子命令返回
`E_GUI_UNAVAILABLE` 并退出码 8；所有无头（headless）能力与测试不受影响。
Debian/Ubuntu 可 `sudo apt-get install python3-tk`；Windows 标准安装自带。
