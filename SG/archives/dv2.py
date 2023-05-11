import pandas as pd
import matplotlib.pyplot as plt
import PySimpleGUI as sg

# 创建 DataFrame
data = {'path': ['file1.txt', 'file2.txt', 'file3.txt', 'file4.txt'],
        'emotion': ['happy', 'sad', 'angry', 'neutral']}
df = pd.DataFrame(data)

# 计算情感成分并按文件名排序
emotion_counts = df['emotion'].value_counts().sort_index()

# 按文件名分组并计算情感成分
grouped = df.groupby(df['path'].str.split('.').str[0])
grouped_counts = grouped['emotion'].value_counts().unstack(fill_value=0)

# 创建窗口布局
layout = [[sg.Text('用户情感信息分析')],
          [sg.Canvas(key='-CANVAS-')],
          [sg.Button('生成情感成分分析图表')],
          [sg.Canvas(key='-CANVAS2-')],
          [sg.Button('生成各文件情感成分分析图表')]]

# 创建窗口
window = sg.Window('用户情感信息分析', layout)

# 定义事件循环
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    elif event == '生成情感成分分析图表':
        # 创建图表
        fig, ax = plt.subplots()
        ax.bar(emotion_counts.index, emotion_counts.values)
        ax.set_title('用户情感成分分析')
        ax.set_xlabel('情感')
        ax.set_ylabel('数量')
        # 将图表绘制到 PySimpleGUI 的 Canvas 中
        canvas = window['-CANVAS-'].TKCanvas
        fig_canvas = fig.canvas
        fig_canvas.draw()
        graph = fig_canvas.get_tk_widget()
        graph.pack(side='top', fill='both', expand=True)
        # 显示图表
        plt.show()
    elif event == '生成各文件情感成分分析图表':
        # 创建图表
        fig, ax = plt.subplots()
        grouped_counts.plot(kind='bar', ax=ax)
        ax.set_title('各文件情感成分分析')
        ax.set_xlabel('文件名')
        ax.set_ylabel('数量')
        # 将图表绘制到 PySimpleGUI 的 Canvas 中
        canvas = window['-CANVAS2-'].TKCanvas
        fig_canvas = fig.canvas
        fig_canvas.draw()
        graph = fig_canvas.get_tk_widget()
        graph.pack(side='top', fill='both', expand=True)
        # 显示图表
        plt.show()

# 关闭窗口
window.close()