import pandas as pd
import matplotlib.pyplot as plt
import PySimpleGUI as sg

# 创建 DataFrame
data = {'path': ['file1.txt', 'file2.txt', 'file3.txt', 'file4.txt'],
        'emotion': ['happy', 'sad', 'angry', 'neutral']}
df = pd.DataFrame(data)

# 计算情感成分
emotion_counts = df['emotion'].value_counts()

# 创建窗口布局
layout = [[sg.Text('情感成分分析图表')],
          [sg.Canvas(key='-CANVAS-')],
          [sg.Button('生成情感成分分析图表')]]

# 创建窗口
window = sg.Window('情感成分分析', layout)

# 定义事件循环
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    elif event == '生成情感成分分析图表':
        # 创建图表
        fig, ax = plt.subplots()
        ax.pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%')
        ax.set_title('情感成分分析')
        # 将图表绘制到 PySimpleGUI 的 Canvas 中
        canvas = window['-CANVAS-'].TKCanvas
        fig_canvas = fig.canvas
        fig_canvas.draw()
        graph = fig_canvas.get_tk_widget()
        graph.pack(side='top', fill='both', expand=True)
        # 显示图表
        plt.show()

# 关闭窗口
window.close()