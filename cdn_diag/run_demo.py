#!/usr/bin/env python3
"""演示入口：python run_demo.py [--inject cycle|truncate|digest]"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from cdn_diag.demo import main

if __name__ == "__main__":
    raise SystemExit(main())
