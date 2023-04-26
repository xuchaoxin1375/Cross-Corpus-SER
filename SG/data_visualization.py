import pandas as pd
import matplotlib.pyplot as plt
import PySimpleGUI as sg

# 创建 DataFrame
data_rows = {
    "path": ["file1.txt", "file2.txt", "file3.txt", "file4.txt"],
    "emotion": ["happy", "sad", "angry", "neutral"],
}
df = pd.DataFrame(data_rows)

# 计算情感成分
emotion_counts = df["emotion"].value_counts()

# 创建窗口布局
layout = [
    [sg.Text("情感成分分析图表")],
    [sg.Canvas(key="-CANVAS-")],
    [sg.Button("generate pie graph")],
]

# 创建窗口
#!注意,客户端的非主源代码文件中的sg.Window()语句要小心放置再安全的地方(比如某个测试函数),否则导致崩溃而且没有错误细节提示
# window = sg.Window("情感成分分析", layout)

from table_show import TableShow
# 定义事件循环
def data_visualize_events(t:TableShow=None, window=None, event=None):
    
    if event == "generate pie graph":
        if t is None:
            print("please select several files from the fviewer frist!😂")
        else:
            data=t.data_df
            print("you trigger the pie graph drawer!")
            # 创建图表
            fig, ax = plt.subplots()
            emo_count=data['emotion']
            from collections import Counter
            counter=Counter(emo_count)
            emo_labels=list(counter.keys())
            x=list(counter.values())
            ax.pie(x=x, labels=emo_labels, autopct="%1.1f%%")
            ax.set_title("emotion composition analyzer")
            # 将图表绘制到 PySimpleGUI 的 Canvas 中
            # canvas = window["-CANVAS-"].TKCanvas
            fig_canvas = fig.canvas
            fig_canvas.draw()
            graph = fig_canvas.get_tk_widget()
            graph.pack(side="top", fill="both", expand=True)
            # 显示图表
            plt.show()


def main_dv(emotion_counts, data_visualize_events):
    # 创建窗口
    window = sg.Window("情感成分分析", layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        data_visualize_events(emotion_counts, window, event)

    # 关闭窗口
    window.close()


if __name__ == "__main__":
    main_dv(emotion_counts, data_visualize_events)
