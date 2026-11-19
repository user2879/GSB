# Windows 本地环境安装说明（Python 3.11）

本工作台仅依赖 Python 标准库、NumPy 与 Pillow，不访问网络即可运行管线
（仅安装依赖时需要联网下载 wheel）。

## 1. 安装 Python 3.11

从 https://www.python.org/downloads/windows/ 下载 Python 3.11.x 安装包，
安装时勾选 **Add python.exe to PATH**。在 PowerShell 中验证：

```powershell
py -3.11 --version
```

## 2. 创建虚拟环境并安装依赖

```powershell
cd path\to\uiddfc5909d3-default-264
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` 锁定本次实际验证使用的版本：

```
numpy==2.2.6
Pillow==11.3.0
```

> 若企业环境无法访问公网，可在有网机器执行
> `pip download -r requirements.txt -d wheels`，拷贝 `wheels/` 到离线机器后
> `pip install --no-index --find-links wheels -r requirements.txt`。

## 3. 验证安装

```powershell
python -c "import sys, numpy, PIL; print(sys.version); print(numpy.__version__, PIL.__version__)"
```

## 4. 生成确定性样本（离线、固定种子）

```powershell
python -m dbench.sample_generators.generate_samples
```

## 5. 运行与测试

```powershell
python -m dbench run --config configs/valid_base.json --store snapshots/demo --report reports/demo.json
python -m unittest discover -s tests -v
```

## 本次实际验证环境记录

| 项 | 值 |
| --- | --- |
| 操作系统 | Ubuntu-22.04 (WSL2, Linux 6.18) |
| Python | 3.11.0rc1 |
| pip | 26.3（通过 get-pip.py 用户级安装） |
| NumPy | 2.2.6 |
| Pillow | 11.3.0 |

代码仅使用跨平台 API（`os.replace` 原子替换、`pathlib`、纯 NumPy 计算、
SHA-256 哈希），在 Windows NTFS 与 Linux 上行为一致；未使用 Docker 或容器。
