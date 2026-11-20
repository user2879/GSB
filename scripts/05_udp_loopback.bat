@echo off
REM 真实 127.0.0.1 回环 UDP 收发链路验证
python scripts\demo.py udp > logs\demo_udp.log 2>&1
type logs\demo_udp.log
