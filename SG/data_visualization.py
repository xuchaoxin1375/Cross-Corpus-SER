##
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from SG.multilanguage import get_language_translator

language = "en"
lang = get_language_translator(language)
# 创建 DataFrame(假设表格内的初始数据是path和emotion两个列),需要自行统计计算
# 各个emotion对应的情感文件数量
data_rows = {
    "path": ["file1.txt", "file2.txt", "file3.txt", "file4.txt"],
    "emotion": ["happy", "sad", "angry", "neutral"],
}
df_demo = pd.DataFrame(data_rows)
# 计算情感成分,这是饼图需要绘制的数据
# emotion_counts = df_demo["emotion"].value_counts()
emotions = data_rows["emotion"]
emotions_counts = Counter(emotions)
category_labels = list(emotions_counts.keys())
category_sizes = list(emotions_counts.values())
##


# 创建窗口布局
def get_dv_layout():
    layout = [
        [sg.Text(lang.emotion_compositon_analyzer_title)],
        [sg.Canvas(key="-CANVAS-")],
        [sg.Button(lang.generate_pie_graph, key="generate pie graph")],
    ]

    return layout


# 创建窗口
#!注意,客户端的非主源代码文件中的sg.Window()语句要小心放置在安全的地方(比如某个测试函数),否则导致崩溃而且没有错误细节提示

import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
def draw_pie_chart(category_sizes, labels, title):

    fig, ax = plt.subplots()
    ax.pie(category_sizes, labels=labels, autopct='%1.1f%%')
    ax.axis('equal')
    # plt.title(title)
    ax.set_title(title)
    plt.show()
    # return fig

# 定义事件循环
def data_visualize_events(emotion_count=None, window=None, event=None):
    from SG.table_show import TableShow
    if event == "generate pie graph":
        if emotion_count is None:
            print("please select several files from the fviewer frist!😂")
            sg.popup(lang.select_audios_prompt)
        else:
            data = emotion_count
            print(type(emotion_count),"@{emotion_count}")
            if isinstance(emotion_count,TableShow):
                print('data: ', data,"emotion_count is instance of TableShow")
            data=emotion_count.data_df
            print("you trigger the pie graph drawer!")

            emotions = data["emotion"]
            counter = Counter(emotions)
            emotion_sizes = list(counter.values())
            emotion_labels = list(counter.keys())
            print("emo_labels: ", emotion_labels)
            # 创建图表
            title = lang.emotion_compositon_analyzer_title
            fig, ax = plt.subplots()
            # 绘制饼图
            draw_pie_chart(emotion_sizes, emotion_labels, title)




def main_dv():
    # 创建窗口
    window = make_window()


    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        data_visualize_events(emotion_count=df_demo, window=window, event=event)

    # 关闭窗口
    window.close()

def make_window():
    layout = get_dv_layout()
    window = sg.Window(lang.emotion_compositon_analyzer_title, layout, finalize=True)
    return window


if __name__ == "__main__":
    main_dv()
