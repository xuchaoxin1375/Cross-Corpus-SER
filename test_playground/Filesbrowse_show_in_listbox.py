import PySimpleGUI as sg

# 设置主题
sg.theme("DarkBlue")

# 定义布局
layout = [
    [sg.Text("选择文件：")],
    [sg.Input(key="-FILES-", enable_events=True, visible=False), sg.FilesBrowse("浏览1")],
    [sg.Text("已选文件：")],
    [sg.Listbox(values=[], size=(50, 10), key="-FILE_LIST-", enable_events=True),sg.FilesBrowse("浏览2")],
    [sg.Button("提交"), sg.Button("取消")],
]

# 创建窗口
window = sg.Window("文件浏览示例", layout)

# 事件循环
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == "取消":
        break
    elif event == "-FILES-":
        files = values["-FILES-"].split(";")  # 分割多个文件名
        window["-FILE_LIST-"].update(files)  # 更新ListBox内容
    elif event == "提交":
        print("选中的文件:", values["-FILE_LIST-"])

# 关闭窗口
window.close()
