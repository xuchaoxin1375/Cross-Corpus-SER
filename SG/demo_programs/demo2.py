import PySimpleGUI as sg

##
# 定义GUI布局
text="demo_Checkbox",
layout = [
    [sg.Checkbox(text, default=True, enable_events=True, key="CB")],
    [sg.Text("AText",key="AText",enable_events=True)],
    [sg.Listbox(["l1","l2","l3","l4"],default_values="l1",size=(10,5),enable_events=True,key="LB")],
    [sg.Button("OK",key="BtnOK")]
]

# 创建窗口并运行事件循环
window = sg.Window("My Window", layout,resizable=True,)
while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED:
        break

    if event is not None:
        print(event, values)
        print()

window.close()
