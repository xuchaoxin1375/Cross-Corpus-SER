from tkinter import *
from tkinter import ttk


def calculate(*args):
    """Performing the Calculation

    As you can clearly see, this routine takes the number of feet from our
    entry widget, does the calculation, and places the result in our label
    widget.
    这里的feet.get()是tkinter中的用法
    feet是一个全局变量(type=StringVar)
    本例中,我需要读取feet的值(用户输入),根据这个值做单位换算,然后通过将计算结果设置到meters中
    读取和设置分别使用StringValue提供的get方法和set方法
    """
    try:
        value = float(feet.get())
        meters.set(int(0.3048 * value * 10000.0 + 0.5) / 10000.0)
    except ValueError:
        pass


# Setting up the Main Application Window
root = Tk()
root.title("Feet to Meters")
# Creating a Content Frame
mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# Creating the Entry Widget
#我们实例化2个可变字符串,它们被渲染在界面上,如果以正确的方式修改它们的值,就可以在界面及时的更新这些值
feet = StringVar()
meters = StringVar()
# Here's where the magic textvariable options we specified
# when creating the widgets come into play. We specified the global
# variable feet as the textvariable for the entry, which means that
# anytime the entry changes, Tk will automatically update the global
# variable feet. 
feet_entry = ttk.Entry(mainframe, width=7, textvariable=feet)
feet_entry.grid(column=2, row=1, sticky=(W, E))

ttk.Label(mainframe, textvariable=meters).grid(column=2, row=2, sticky=(W, E))
ttk.Button(mainframe, text="Calculate", command=calculate).grid(
    column=3, row=3, sticky=W
)

ttk.Label(mainframe, text="feet").grid(column=3, row=1, sticky=W)
ttk.Label(mainframe, text="is equivalent to").grid(column=1, row=2, sticky=E)
ttk.Label(mainframe, text="meters").grid(column=3, row=2, sticky=W)


# adding some polish
for child in mainframe.winfo_children():
    child.grid_configure(padx=5, pady=5)

feet_entry.focus()
root.bind("<Return>", calculate)
# Start the Event Loop
root.mainloop()

##
from tkinter import *
import tkinter as tk

root = Tk()
# 设置主窗口大小
root.geometry("200x100")
#基于主窗口直接实例一个文本标签控件
label = Label(root, text="Hello, World!")
#将控件安排到窗口的某个位置
label.grid(row=0, column=0, sticky=(tk.W, tk.E))

root.mainloop()
##
import tkinter as tk

root = tk.Tk()

frame = tk.Frame(root, bg="gray", padx=10, pady=10)
frame.grid(row=0, column=0, sticky="NESW")

btn1 = tk.Button(frame, text="Button 1")
btn1.grid(row=0, column=0, sticky="W")

btn2 = tk.Button(frame, text="Button 2")
btn2.grid(row=1, column=0, sticky="E")

btn3 = tk.Button(frame, text="Button 3")
btn3.grid(row=0, column=1, rowspan=2, sticky="NS")

root.mainloop()
##
import tkinter as tk

root = tk.Tk()

frame = tk.Frame(root, bg="lightgray", padx=10, pady=10)
frame.grid(row=0, column=0)

btn1 = tk.Button(frame, text="Button 1")
btn1.grid(row=0, column=0, sticky="NW")

btn2 = tk.Button(frame, text="Button 2")
btn2.grid(row=1, column=0, sticky="SE")

btn3 = tk.Button(frame, text="Button 3")
btn3.grid(row=0, column=1, rowspan=2, sticky="NS")

root.mainloop()