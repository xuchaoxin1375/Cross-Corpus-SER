##
import PySimpleGUI as sg
from audio.core import get_used_keys
from config.EF import ava_algorithms
from config.MetaPath import ava_dbs
from audio.graph import showWaveForm, showMelFreqGraph, showFreqGraph

##

train_db = ""
test_db = ""
size = (1000, 1000)
# size=None


def db_layout(set=""):
    db_list_layout = [
        [sg.Button(db, tooltip=f"set the training data base as {db}")] for db in ava_dbs
    ]
    # format(db,"^10")
    db_label = [sg.T(f"chose the db for {set}:")]
    db_layout = [db_label, db_list_layout]
    return db_layout


def choose_db(window, db_config):
    while True:
        event, value = window.read()
        # print(event,type(event),len(event))
        db_config = []
        if event:
            db_config.append(event)
        print(event, value)
        # permenent
        if event in (sg.WIN_CLOSED, "Quit"):
            break
    print(db_config, "@{db_config}")


# dbs_layout = [db_layout("train"), db_layout("test")]
# main_layout = [[dbs_layout, [sg.B("Quit")]]]

# layout=main_layout
train = "trian"
test = "test"

db_choose_layout = [
    [sg.Text("Select the training database")],
    [sg.Combo(ava_dbs, key="train_db")],
    [sg.Text("Select the testing database")],
    [sg.Combo(ava_dbs, key="test_db")],
    [sg.Button("OK"), sg.Button("Cancel")],
]

algos = []
for algo in ava_algorithms:
    algos.append(sg.Radio(algo.upper(), "algorithm", key=f'{algo}'))

algos_layout = [
    [
        sg.Text(
            "选择一个算法进行试验:"
        )
    ],
    algos,
    [sg.Button('OK'), sg.Button('Cancel')]
]

features_choose_layout = [
    [sg.Text("请选择一个或多个特征：")],
    [
        sg.Checkbox("MFCC", key="mfcc"),
        sg.Checkbox("Mel", key="mel"),
        sg.Checkbox("Contrast", key="contrast"),
    ],
    [sg.Checkbox("Chromagram", key="chroma"), sg.Checkbox("Tonnez", key="tonnez")],
    [sg.Button("确定"), sg.Button("取消")],
]
# file_choose_layout = [
#     [sg.Text("请选择一个音频文件样本,识别其情感")],
#     [sg.Input(), sg.FileBrowse()],

#     # [sg.Button('确定'), sg.Button('取消')]
# ]

file_choose_layout = [
    [sg.Text("请选择一个音频文件样本,识别其情感")],
    [
        sg.Combo(
            sorted(sg.user_settings_get_entry("-filenames-", [])),
            default_value=sg.user_settings_get_entry("-last filename-", ""),
            size=(50, 1),
            key="-FILENAME-",
        ),
        sg.FileBrowse(),
        sg.B("Clear History"),
    ],
    [sg.Button("Ok", bind_return_key=True), sg.Button("Cancel")],
]

draw_layout = [
    [sg.Text("请选择一个文件,绘制其[波形图|频谱图|Mel频谱图]：")],
    [sg.Input(), sg.FileBrowse()],
    [
        sg.Checkbox("waveForm", key="wave_form"),
        sg.Checkbox("FreqGraph", key="freq_graph"),
        sg.Checkbox("MelFreqGraph", key="mel_freq_graph"),
        [sg.Button("确定"), sg.Button("取消")],
    ],
]

# [sg.Checkbox(emo) for emo in ava_emotions]
emotions_layout =[
    [sg.Text("choose the emotion config："),
     sg.Text(
            "请选择一个情感组合进行试验：推荐组合AS,HNS,AHNS,AHNPS\n\
             注意,savee库种的`surprise`和`pleasantSurprise`)有一定区别,所以AHNPS组合不推荐用于savee上"
        )],
    [
        sg.Checkbox("angry",key='angry'),
        sg.Checkbox("happy",key='happy'),
        sg.Checkbox("neutral",key='neutral'),
        sg.Checkbox("pleasantSuprise",key='pleasantSuprise'),
        sg.Checkbox("sad",key='sad')
    ],
    [sg.Button("OK"), sg.Button("Cancel")],
]

old_db_choose_layout = [
    [sg.T(f"chose the db for {set}:") for set in (train, test)],
    [
        sg.LB(values=ava_dbs, size=(15, 5), key=f"-train_list-"),
        sg.LB(values=ava_dbs, size=(15, 5), key=f"-test_list-"),
    ],
]

layout = [
    db_choose_layout,

    emotions_layout,
    algos_layout,
    features_choose_layout,
    file_choose_layout,
    draw_layout,
    [sg.B("OK"), sg.B("Quit")],
]
window = sg.Window(title="ccser_client", layout=layout, alpha_channel=0.9, size=size)

##
# while True:
# event,values=window.read()
# # if event in (sg.WIN_CLOSED,"Cancel"):
# train_db=values['train_db']
# test_db=values['test_db']

# Initialize the selected databases list
selected_databases = []

#语料库的选择
# Loop until the user clicks the OK button or closes the dialog
while True:
    event, values = window.read()

    train_db=values['train_db']
    test_db=values['test_db']
    print(train_db, "@{trian_db}")
    print(test_db, "@{test_db}")
    # print(values)
    if train_db and test_db:
        break

    if event in (None,"OK", "Cancel"):
        break
    # Add the selected databases to the list
    # selected_databases.append(values["train_db"])
    # selected_databases.append(values["test_db"])
    # If the user has selected two databases, exit the loop
    # if len(selected_databases) == 2:
    #     break
print("已完成语料库的选择.")

# print(train_db, "@{trian_db}")
# print(test_db, "@{test_db}")
e_config=[]
while True:
    event, values = window.read()
    e_config_dict=dict(
        angry=values['angry'],
        happy=values['happy'],
        neutral=values['neutral'],
        pleasantSuprise=values['pleasantSuprise'],
        sad=values['sad']
    )
    for emo in  get_used_keys(e_config_dict):
        if emo:
            e_config.append(emo)
    print(e_config,"@{e_config}")
    if event in (None, "OK", "Cancel"):
        break
print("完成情感组合的选择.")

algo
while True:
    event, values = window.read()
    algo=values['algo']
    if event in (None, "OK", "Cancel"):
        break
print("完成算法的选择.")


# window = sg.Window("Filename Chooser With History", layout)
#这部分只负责选取文件,选取通过点击确认,来完成这部分逻辑,跳到循环,执行下一步分代码
while True:
    event, values = window.read()

    if event in (sg.WIN_CLOSED, "Cancel"):
        break
    if event == "Ok":
        # If OK, then need to add the filename to the list of files and also set as the last used filename
        sg.user_settings_set_entry(
            "-filenames-",
            list(
                set(
                    sg.user_settings_get_entry("-filenames-", [])
                    + [
                        values["-FILENAME-"],
                    ]
                )
            ),
        )
        sg.user_settings_set_entry("-last filename-", values["-FILENAME-"])
        # 打印事件和此时此刻key='-FILENAME-'的(也就式文件名的)输入式元素的值
        audio_selected=values['-FILENAME-']
        print(event, values['-FILENAME-'])
        break
    
    elif event == "Clear History":
        sg.user_settings_set_entry("-filenames-", [])
        sg.user_settings_set_entry("-last filename-", "")
        window["-FILENAME-"].update(values=[], value="")
audio_selected=audio_selected
print("完成文件选取")
# 读取checkbox的输入(目前的写法需要先点击前面的表单,后面的表单才可以正确响应)
while True:
    event, values = window.read()
    if not event:
        break
    wave_form=values['wave_form']
    freq_graph=values['freq_graph']
    mel_freq_graph=values['mel_freq_graph']
    print(f"{event=}in draw tasks..(开始绘制.)")
    if wave_form:
        showWaveForm(audio_selected)
    if freq_graph:
        showFreqGraph(audio_selected)
    if mel_freq_graph:
        showMelFreqGraph(audio_selected)
    if event in (sg.WIN_CLOSED, "OK","Cancel"):
        break
print("完成图形绘制.")
while True:
    event, value = window.read()
    # print(event, value)
    # if event:
    #     # train_db=value['-train_list-']
    #     # print(value['-train_list-'],"@value['-train_list-']")
    #     # test_db=value['-test_list-']
    #     # print(event, value)
    if event in (sg.WIN_CLOSED, "OK", "Quit"):
        break
print("关闭窗口.")


window.close()
