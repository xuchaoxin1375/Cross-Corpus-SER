import PySimpleGUI as sg
import json
from config.MetaPath import trans_en_json,trans_zh_json
from SG.translations import en,zh

# lang\["(.+)"\]
#lang.$1
lang=en

English_marks = ('English','英文','en')
Chinese_marks = ('中文',"Chinese",'zh','cn')

def get_language_translator(language="English"):
    global lang
    if language in English_marks:
        lang=en
        # return en
        # return zh
    elif language in Chinese_marks:
        lang=zh
    return lang
    
def get_language_translator_json(language="Chinese"):
    if language in English_marks:
        with open(trans_en_json, 'r',encoding='utf-8') as f:
            translations = json.load(f)
    elif language in Chinese_marks:
                # encoding='utf-8',否则中文字符报错:UnicodeDecodeError: 'charmap' codec can't decode byte 0x81 in position 53: character maps to <undefined>
        with open(trans_zh_json, 'r',encoding='utf-8') as f:
            translations = json.load(f)
    lang=translations
    return lang




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
            translations = get_language_translator(values['language'])
            print(translations,values)
            # 显示欢迎消息
            # welcome_message = translations['welcome_message']
            welcome_message = translations.welcome_message
            window['welcome_message'].update(welcome_message)

    # 关闭 GUI 窗口
    window.close()


if __name__ == '__main__':
    # res=get_your_language_translator('中文')
    # print('res: ', res)
    run_app()
    