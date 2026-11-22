#!/usr/bin/env python3
"""诊断台 GUI 入口（需要 tkinter，Windows 版 Python 自带）。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from cdn_diag.gui import main

if __name__ == "__main__":
    main()
