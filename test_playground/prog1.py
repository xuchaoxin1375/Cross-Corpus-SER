# prog1.py
import argparse

# 创建一个ArgumentParser对象
parser = argparse.ArgumentParser()

# 添加一个可选参数
parser.add_argument("--verbose", "-v", action="store_true", help="verbose mode")

# 解析命令行参数
args = parser.parse_args()


# 打印可选参数的值
if args.verbose:
    print("Verbose mode is on")
else:
    print("Verbose mode is off")