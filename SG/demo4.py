# import PySimpleGUI as sg

# # 创建文件浏览框
# layout = [[sg.Text('请选择一个文件夹：')], [sg.In(key='-FOLDER-'), sg.FolderBrowse(enable_events=True)]]

# # 显示窗口并运行事件循环
# window = sg.Window('文件浏览框', layout)
# event,values="1","1"
# while True:
#     event, values = window.read()
#     print(event,values)
#     if event == sg.WINDOW_CLOSED:
#         break
#     else:
#         folder = values['-FOLDER-']
#         sg.popup(f'你选择的文件夹是：{folder}')
#     print(event,values)
# print(event,values)
# window.close()
# import PySimpleGUI as sg

# layout = [
#     [sg.Input(key='-FILE-', enable_events=True), sg.FileBrowse()],
#     [sg.Button('OK'), sg.Button('Cancel')]
# ]

# window = sg.Window('File Browse', layout)

# while True:
#     event, values = window.read()
#     if event in (sg.WIN_CLOSED, 'Cancel'):
#         break
#     elif event == '-FILE-':
#         print(f'File selected: {values["-FILE-"]}')

# window.close()
