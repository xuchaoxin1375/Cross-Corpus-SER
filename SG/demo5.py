import PySimpleGUI as sg

layout = [
    [sg.T("Source Folder")],
    [sg.In(key="input")],
    [
        sg.FolderBrowse(
            target="input",  # 这里显式的将选取结果通过key的方式绑定到上面的文本框(`input`)
            enable_events=True,
            change_submits=True,
        ),
        sg.OK(),
    ],
    [
        sg.Input(
            default_text="select multiple files,which will be shown here ",
            key="files input",
        ),
        sg.FilesBrowse(target="files input", enable_events=True, change_submits=True),
    ],
]
while True:
    window = sg.Window("My App", layout, resizable=True)
    event, values = window.read()
    # 当文件选择完成后,依然会处于阻塞状态,需要借助其他键来继续(比如这里的OK)
    # IMPORT NOTE ABOUT SHORTCUT BUTTONS Prior to release 3.11.0, these buttons closed the window.
    # Starting with 3.11 they will not close the window. They act like RButtons (return the button text and do not close the window)
    if event:
        print(event, values)
    if event in (sg.WIN_CLOSED, "OK"):
        break

window.close()
##
import PySimpleGUI as sg

layout = [[sg.Text('Select a file:')],
          [sg.Input(key='-FILE-'), sg.FileBrowse()],
          [sg.Button('Submit'), sg.Button('Cancel')]]

window = sg.Window('File Browser', layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel':
        break
    elif event == 'Submit':
        file_path = values['-FILE-']
        # 处理文件路径的代码
        print(file_path,event)

window.close()
