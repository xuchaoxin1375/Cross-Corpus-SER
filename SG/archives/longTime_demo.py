import csv
import threading
import time
import PySimpleGUI as sg

# 定义耗时操作
def long_operation():
    # 模拟耗时操作
    time.sleep(5)

# 定义写入 CSV 文件的函数
def write_csv():
    # 模拟写入数据
    data = [['John', 'Doe', 25], ['Jane', 'Doe', 30], ['Bob', 'Smith', 45]]
    with open('data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

# 创建窗口布局
layout = [[sg.Button('写入数据到CSV文件')]]

# 创建窗口
window = sg.Window('进度条示例', layout)

# 定义事件循环函数
def event_loop():
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        elif event == '写入数据到CSV文件':
            # 创建新线程来执行耗时操作
            thread = threading.Thread(target=write_csv)
            thread.start()
            # 模拟进度
            for i in range(100):
                time.sleep(0.05)
                window['-PROGRESS-'].UpdateBar(i+1)

# 创建进度条并添加到窗口布局中
progress_bar = sg.ProgressBar(100, orientation='h', size=(20, 20), key='-PROGRESS-')
layout.append([progress_bar])

# 启动事件循环
event_loop()

# 关闭窗口
window.close()