# 读取工作区相对输入文件，输出确定性摘要信息
import io
def main():
    with io.open("files/data.csv", "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    print("rows=" + str(len(lines)))
    for line in lines:
        print("field:" + line.split("=", 1)[0])
if __name__ == "__main__":
    main()
