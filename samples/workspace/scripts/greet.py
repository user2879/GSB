import sys
# 确定性输出：仅依赖固定输入，不读取时钟、不访问网络
name = sys.argv[1] if len(sys.argv) > 1 else "world"
print(f"hello {name}")
