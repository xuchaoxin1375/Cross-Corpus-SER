import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def draw_pie_chart(category_sizes, labels, title):
    fig, ax = plt.subplots()
    ax.pie(category_sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    plt.title(title)
    return fig

def main():
    # 饼图数据和标签
    sizes = [30, 20, 25, 25]
    labels = ['A', 'B', 'C', 'D']
    title = "Example Pie Chart"

    # 绘制饼图并获取图形对象
    fig = draw_pie_chart(sizes, labels, title)

    # 创建一个用于显示饼图的窗口
    layout = [[sg.Canvas(size=(800, 600), key='-CANVAS-')],
              [sg.Button('OK', bind_return_key=False)]]

    window = sg.Window('Pie Chart Example', layout, finalize=True)

    # 将饼图添加到窗口的Canvas上
    canvas_elem = window['-CANVAS-']
    canvas = FigureCanvasTkAgg(fig, master=canvas_elem.TKCanvas)
    canvas.draw()
    canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

    # 显示窗口并处理事件
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'OK':
            break

    window.close()

if __name__ == '__main__':
    main()