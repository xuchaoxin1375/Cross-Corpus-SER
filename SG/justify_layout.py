import PySimpleGUI as sg

# 创建两个按钮
button_first = sg.Button("Button Frist")
button_m1 = sg.Button("Button m1")
button_m2 = sg.Button("Button m2")
button_last = sg.Button("Button Last")

# 创建一个Frame，包含两个垂直布局的列
#技巧在于,使用sg.Column打包需要被布局的元素,最后一个的expend_x=False
frame_layout = [
    [
        sg.Column([[button_first]], justification="left", expand_x=True),
        sg.Column([[button_m1]], justification="center", expand_x=True),
        sg.Column([[button_m2]], justification="center", expand_x=True),
        sg.Column([[button_last]], justification="right", expand_x=False),
    ]
]

# 将Frame添加到窗口布局中
layout = [[sg.Frame("My Frame", frame_layout, expand_x=True)]]

# 创建并显示窗口
window = sg.Window("Align Buttons in Frame", layout, resizable=True, finalize=True)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break

window.close()
