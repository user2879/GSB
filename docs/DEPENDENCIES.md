# 环境与依赖安装（Linux / WSL Ubuntu）

## 已确认环境

- Ubuntu 22.04.5 LTS（WSL）。
- 默认验收运行时：Python 3.11.0rc1（路径可能由反馈环境提供）；代码只使用 Python 3.10+ 语法。
- 已验证 Python 3.11 可导入 NumPy 1.26.4 与 PyYAML。
- 核心渲染、PNG 编解码、快照和测试不依赖 Tkinter、Docker、GPU、网络服务或付费资源。

## 推荐系统级安装

在普通 WSL Ubuntu 终端执行：

```bash
sudo apt update
sudo apt install -y \
  python3 python3-dev python3-pip python3-venv \
  python3-tk python3-numpy python3-yaml

python3 -m pip install -r requirements.txt
python3 -c "import sys, numpy, yaml, tkinter; print(sys.version); print(numpy.__version__, yaml.__version__, tkinter.TkVersion)"
```

也可以直接运行：

```bash
bash scripts/install_deps.sh
```

## 无 root / 当前定制 Python 缺少 libtk8.6.so

本工作区已用免 root 方式下载并解包以下 Ubuntu deb 到 `vendor/tk/`：

- `libtcl8.6`
- `libtk8.6`
- `tcl8.6`
- `tk8.6`
- `libxft2`
- `libxss1`

重新获取可运行：

```bash
bash scripts/fetch_local_tk_vendor.sh
source scripts/tk_vendor_env.sh
python3 -m condrender.cli gui --root .
```

`source scripts/tk_vendor_env.sh` 会设置 `LD_LIBRARY_PATH`、`TCL_LIBRARY` 和 `TK_LIBRARY`。在无显示的纯终端中，`tkinter` 可以导入，但创建根窗口会报 `DISPLAY` 错误；核心 CLI 与测试不受影响。WSLg 或 X Server 可用时可启动诊断面板。

## 可选虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

若使用自己编译/额外安装的 Python 3.11，`apt install python3-tk` 的扩展可能只安装到系统 Python 3.10；此时优先使用 `/usr/bin/python3`（Ubuntu 22.04 为 3.10），或按上面的免 root vendor 方式补齐共享库。
