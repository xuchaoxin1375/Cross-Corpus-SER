# stdin_readline_demo.py
# Python program to demonstrate
# sys.stdin.readline()


import sys
print("read string:")
name = sys.stdin.readline()
print(repr(name),type(name))

#可以控制读入的字符数
print("read 2 digit:")
num = sys.stdin.readline(2)
print(num,type(num))
# 比较input()
line=input("read by input():")
print(repr(line),type(line))
