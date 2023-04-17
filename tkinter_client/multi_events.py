import tkinter as tk

# 创建一个 Tkinter 窗口
root = tk.Tk()

# 定义一个事件处理函数
def handle_event_position_info(event):
    print(f'Event {event.type} occurred at ({event.x}, {event.y})')

def update_text(w):
    print("right button of mouse clicked!")
# 创建一个 Canvas 组件
canvas = tk.Canvas(root, width=200, height=200)

# 向 Canvas 组件绑定多个事件
canvas.bind('<Button-1>', handle_event_position_info)
canvas.bind('<Button-2>', handle_event_position_info)
canvas.bind('<Button-3>', update_text)
canvas.bind('<ab>', lambda x:print("KeyPress-A and KeyPress-B event happend"))

# 将 Canvas 组件添加到窗口中
canvas.pack()

# 进入 Tkinter 主循环
root.mainloop()
