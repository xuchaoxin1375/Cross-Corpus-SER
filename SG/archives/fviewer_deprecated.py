import os
import PySimpleGUI as sg

# 创建GUI窗口
sg.theme('DarkBlue')
layout = [
    [sg.Text('Select a directory:'), sg.InputText(), sg.FolderBrowse()],
    [sg.Checkbox('Recursively scan subdirectories', default=False, key='recursive_checkbox')],
    [sg.Button('Filter'), sg.Button('Exit')],
    [sg.Text('Filtered audio files:')],
    [sg.Listbox(values=[], size=(50, 10), key='audio_files_list', enable_events=True)],
    [sg.Button('Confirm'), sg.Button('Cancel')],
    [sg.Text('Selected audio files:')],
    [sg.Listbox(values=[], size=(50, 10), key='selected_files_list')],
]
window = sg.Window('Audio File Filter', layout)

# 事件循环
while True:
    event, values = window.read()

    # 处理事件
    if event in (sg.WINDOW_CLOSED, 'Exit'):
        break
    elif event == 'Filter':
        directory = values[0]
        recursive = values['recursive_checkbox']
        audio_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(('.mp3', '.wav', '.ogg')):
                    audio_files.append(os.path.join(root, file))
                    if not recursive:
                        break
            if not recursive:
                break
        if len(audio_files) == 0:
            sg.popup('No audio files found in the selected directory!')
            window['audio_files_list'].Update(values=[])
        else:
            window['audio_files_list'].Update(values=audio_files)
    elif event == 'audio_files_list':
        selected_files = values['audio_files_list']
        window['selected_files_list'].Update(values=selected_files)
    elif event == 'Confirm':
        selected_files = values['selected_files_list']
        if len(selected_files) == 0:
            sg.popup('Please select at least one audio file!')
        else:
            confirm_message = '\n'.join(selected_files)
            if sg.popup_yes_no(f'Confirm the following files?\n{confirm_message}') == 'Yes':
                # 在这里执行确认操作
                sg.popup('Files confirmed!')
                window['selected_files_list'].Update(values=[])
                window['audio_files_list'].Update(values=[])
    elif event == 'Cancel':
        window['selected_files_list'].Update(values=[])

# 关闭GUI窗口
window.close()