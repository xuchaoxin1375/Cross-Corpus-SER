import PySimpleGUI as sg

##
# 定义GUI布局
text = ("demo_Checkbox",)
# layout = [
#     [sg.Checkbox(text, default=True, enable_events=True, key="CB")],
#     [sg.Text("AText",key="AText",enable_events=True)],
#     [sg.Listbox(["l1","l2","l3","l4"],default_values="l1",size=(10,5),enable_events=True,key="LB")],
#     [sg.Button("OK",key="BtnOK")]
# ]
layout = [
    [
        sg.Text(
            "See how elements look under different themes by choosing a different theme here!"
        )
    ],
    [
        sg.Listbox(
            values=sg.theme_list(),
            size=(20, 12),
            key="-THEME LISTBOX-",
            enable_events=True,
        )
    ],
    [sg.Button("Set Theme")],
]


# 创建窗口并运行事件循环
window = sg.Window(
    "My Window",
    layout,
    resizable=True,
)
while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED:
        break

    if event is not None:
        print(event, values)
        print()

window.close()
