# prog0.py
import argparse

# 创建一个ArgumentParser对象
parser = argparse.ArgumentParser()
print(f'{parser=},{type(parser)=}')
# type(parser)=<class 'argparse.ArgumentParser'>
# 添加一个可选参数
parser.add_argument("--verbose", "-v", action="store_true", help="verbose mode")

# 解析命令行参数
args = parser.parse_args()
print(f'{args=},{type(args)=}')
# type(args)=<class 'argparse.Namespace'>
# ArgumentParser对象经过parse_args()解析之后,得到的args时Namespace对象,它可以通过args.verbose的方式访问parser.add_argument方法添加的参数


# 打印可选参数的值
if args.verbose:
    print("Verbose mode is on")
else:
    print("Verbose mode is off")
