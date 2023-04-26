# stdin_out.py
import sys

print("使用stdin.readline()读取输入(从命令行读取输入):")
input_str = sys.stdin.readline()
print("<<<<<<<<<<")
print("使用repr检查换行符等转义字符",repr(input_str))
print("使用sys.stdout.write()将内容写出到终端:")
sys.stdout.write(repr(input_str))#不同于print()不会追加换行
print("<<<<<<<<<<")


