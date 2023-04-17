##
import tkinter as tk
tk._test()
tk.Tcl().eval('info patchlevel')
##
from tkinter import * 
from tkinter import ttk 
root = Tk() 
btn = ttk.Button(root, text="Hello World")
btn.grid() 
root.mainloop()
##
# 导入Tkinter模块:
import tkinter as tk
# 创建主窗口:
root = tk.Tk()
# 添加窗口元素:
label = tk.Label(root, text="Hello, World!")
label.pack()
# 运行主循环:
root.mainloop()
# 完整代码如下：

##
from tkinter import *
from tkinter import ttk

root = Tk()

frm = ttk.Frame(root, padding=10)

label=ttk.Label(frm, text="Hello World!")
frm.grid()
label.grid(column=0, row=0)

btn=ttk.Button(frm, text="Quit", command=root.destroy)
btn.grid(column=1, row=0)

root.mainloop()

##
import tkinter as tk
import tkinter.ttk as ttk

root = tk.Tk()
frm = ttk.Frame(root)
# frm.pack()

btn = ttk.Button(frm,text="this is a button")
# btn.pack()

keys=btn.configure().keys()
print(keys)#type:ignore
# print(dir(btn))
print()
print(set(dir(btn)) - set(dir(frm)),"@{set(dir(btn)) - set(dir(frm))}")
##
print(set(btn.configure().keys()) - set(frm.configure().keys()))
# root.mainloop()
##
import tkinter as tk

root = tk.Tk()

label1 = tk.Label(root, text="Label 1", bg="red")
label1.pack(side="left")

label2 = tk.Label(root, text="Label 2", bg="green")
label2.pack(side="left")

label3 = tk.Label(root, text="Label 3", bg="blue")
label3.pack(side="left",fill=tk.Y,expand=1)

root.mainloop()
##
import tkinter as tk

root = tk.Tk()

label1 = tk.Label(root, text="Label 1", bg="red")
label1.grid(row=0, column=0)

label2 = tk.Label(root, text="Label 2", bg="green")
label2.grid(row=0, column=1)

label3 = tk.Label(root, text="Label 3", bg="blue")
label3.grid(row=1, column=0, columnspan=2)

root.mainloop()
##
import tkinter as tk

root = tk.Tk()

fred = tk.Label(root, text="Hello, world!", relief="raised")
fred.pack()

print(fred.config())

root.mainloop()
##
from tkinter import ttk 
from tkinter import *
root=Tk()
root.geometry("300x300")
frame=ttk.Frame(root)
frame.grid()
label=ttk.Label(frame,text="Hello World!")
label.grid()


def introspect_hierarchy(w, depth=0):
    """
    使用DFS的策略递归检查参数w的信息
    Prints the class, width, height, x and y coordinates of a tkinter widget and
    its children recursively, with indentation to indicate hierarchy depth.

    Args:
        w (tkinter widget): The widget to print information about.
        depth (int): The depth of the widget in the hierarchy. Default is 0.
    """
    indent = "  " * depth
    info = f"{w.winfo_class()} w={w.winfo_width()} h={w.winfo_height()} x={w.winfo_x()} y={w.winfo_y()} rootx={w.winfo_rootx()} rooty={w.winfo_rooty()}"

    print(f"{indent}{info}")

    for child in w.winfo_children():
        introspect_hierarchy(child, depth + 1)

introspect_hierarchy(root)
# print_hierarchy(frame)
# print_hierarchy(label)
root.mainloop()

##
from tkinter import * 
from tkinter import ttk 
root = Tk() 
tip="Starting..."
label =ttk.Label(root, text=tip) 
label.grid() 
#binding event by bind api:
#鼠标事件移动事件
label.bind('<Enter>', lambda e: label.configure(text='Moved mouse inside')) 
label.bind('<Leave>', lambda e: label.configure(text='Moved mouse outside')) 
#鼠标点击事件
label.bind('<ButtonPress-1>', lambda e: label.configure(text='Clicked left mouse button')) 
#这里<ButtonPress>表示事件名称
#-1(hyphen1)后缀表示鼠标主键(左键)
#可以<ButtonPress-n>简写为<n>
#<2>鼠标中键(滚轮键重压)(不常用)
label.bind('<2>', lambda e: label.configure(text='Clicked middle mouse button')) 
#<3>等价于<ButtonPress-3>也等价于<Button-3>,也就是鼠标右键事件
label.bind('<3>', lambda e: label.configure(text='Clicked right ouse button')) 
# <Double-1>是<Double-ButtonPress-1>的缩写,表示双击鼠标左键
label.bind('<Double-1>', lambda e: label.configure(text='Double clicked')) 
# 鼠标长按并右键拖动,显示坐标
#这个事件包含了鼠标按压联合移动,同时演示了时间参数(e.x,e.y)的使用(鼠标移动返回坐标)
#这里<B3>鼠标右键(ButtonPress-3),Motion是对鼠标移动的捕获

label.bind('<B3-Motion>', lambda e: label.configure(text='right button drag to %d,%d' % (e.x, e.y))) 
#启动循环事件
root.mainloop()

##
import tkinter as tk

def func1(event):
    print("function 1")

def func2(event):
    print("function 2")

root = tk.Tk()
button = tk.Button(root, text="Click me")
button.grid()

button.bind("<Button-1>", func1)
button.bind("<Button-1>", func2,add=True)

root.mainloop()