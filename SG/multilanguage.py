import PySimpleGUI as sg
import json



def get_your_language_translator(language="English"):
    if language in ('English','英文'):
        with open('translations/en.json', 'r',encoding='utf-8') as f:
            translations = json.load(f)
    elif language in ('中文',"Chinese"):
                # encoding='utf-8',否则中文字符报错:UnicodeDecodeError: 'charmap' codec can't decode byte 0x81 in position 53: character maps to <undefined>
        with open('translations/zh.json', 'r',encoding='utf-8') as f:
            translations = json.load(f)
    return translations




def run_app():
    
    # 创建 GUI 窗口
    layout = [[sg.Text('check your language：')],
              [sg.Combo(['English', '中文'], key='language')],
              [sg.Text('<some text to be refresh>', size=(20, 1), key='welcome_message')],
              [sg.Button('OK'), sg.Button('Cancel')]]
    window = sg.Window('My App', layout)

    # 循环处理事件
    while True:
        event, values = window.read()

        # 处理事件
        if event in (sg.WIN_CLOSED, 'Cancel'):
            break
        elif event == 'OK':
            # 根据用户的选择读取相应的语言翻译文件
            translations = get_your_language_translator(values)

            # 显示欢迎消息
            welcome_message = translations['welcome_message']
            window['welcome_message'].update(welcome_message)

    # 关闭 GUI 窗口
    window.close()


if __name__ == '__main__':
    run_app()