##
import PySimpleGUI as sg
from joblib import load

from audio.core import get_used_keys
from audio.graph import showFreqGraph, showMelFreqGraph, showWaveForm
from config.EF import ava_algorithms, ava_emotions, ava_features
from config.MetaPath import (
    ava_dbs,
    bclf,
    brgr,
    emodb,
    get_example_audio_file,
    ravdess,
    savee,
)
from recognizer.basic import EmotionRecognizer
from recognizer.deep import DeepEmotionRecognizer
import sys


def import_config_bookmark():
    pass


##


size = (1500, 1000)
# size=None

# layout=main_layout
train = "trian"
test = "test"


##
def get_algos_elements_list():
    """
      Radio组件需要设置分组,这里分组就设置额为:algorithm
    利用default参数设置选中默认值(默认选中第一个)
    为例在循环中实现设置第i个选项的默认值,可以使用index或者key-value,判断当用一个bool表达式作为default值

    Returns
    -------
    _type_
        _description_
    """
    algos_radios = []
    for i, algo in enumerate(ava_algorithms):
        algos_radios.append(
            sg.Radio(
                algo.title(),
                "algorithm",
                key=f"{algo}",
                default=(i == 1),
                enable_events=True,
            )
        )

    return algos_radios


def create_border_frame(result="inputYourContentToHighligt", key="border"):
    """创建一个带边框的布局窗口

    Parameters
    ----------
    key : str, optional
        _description_, by default "border"

    examples:
    -
        # 创建一个使用border_frame的布局
        demo_border_layout = [
            [sg.Text("Enter a number:"), sg.Input(key="-NUMBER-")],
            [sg.Button("Calculate"), sg.Button("Exit")],
            [create_border_frame(**kwargs)]
        ]

    Returns
    -------
        layout

    """
    # 创建一个带边框区域
    res_layout = [
        [
            sg.Text(
                f"{result}",
                font=("Helvetica", 24, "bold"),
                background_color="green",
                text_color="red",
                key=f"{key}",
            )
        ],
        [sg.HorizontalSeparator()],
        # [sg.Text("Result: "), sg.Text("", size=(20, 1),)]
    ]

    frame = sg.Frame(
        "Result Area",
        res_layout,
        relief=sg.RELIEF_SUNKEN,
        border_width=2,
    )

    return frame


##
# ---create the window---
def make_window(theme=None, size=None):
    if theme:
        sg.theme(theme)
    menu_def = [["&Application", ["E&xit"]], ["&Help", ["&About"]]]
    # 据我观察,通常布局的类型为list[list[element]],也就是说,是一个关于sg组件元素的二轴数组布局,不妨称之为基础布局
    # 并且,若我们将排放在同一行的元素,(称他们为一个元素序列),元素序列的包含sg.<element>个数可以是>=1的
    # 从这个角度理解,那么布局可以理解为`元素序列`按照shape=(-1,1)的形状排放
    # 尽管有这样的嵌套约束,但是PySimpleGui提供了一些用于嵌套的组件,例如sg.Column
    # 我们可以基础布局作为Column组件的参数,然后我可以将Column作为组件放到一个新的基础组件中,这样就好像嵌套一个更深的层布局
    # 在实践中,比较少用过度变量,但是用来作为划分(设计)用途还是不错的,甚至设计完毕后可以销毁这些临时子布局变量

    # ---choose theme---
    theme_layout = [
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
    # ---create 2 column layout---
    # ---column left---
    db_choose_layout = [
        [sg.Text("Select the training database")],
        [sg.Combo(ava_dbs, key="train_db", default_value=emodb, enable_events=True)],
        [sg.Text("Select the testing database")],
        [sg.Combo(ava_dbs, key="test_db", default_value=savee, enable_events=True)],
        # [sg.Button("OK0"), sg.Button("Cancel")],
    ]  # shape=(-1,1)

    # [sg.Checkbox(emo) for emo in ava_emotions]
    e_config_layout = [
        [
            sg.Text("choose the emotion config："),
        ],
        [
            sg.Text(
                "请选择一个情感组合进行试验：推荐组合AS,HNS,AHNS,AHNPS\n\
             注意,savee库种的`surprise`和`pleasantSurprise`)有一定区别,\n所以AHNPS组合不推荐用于savee上"
            )
        ],
        [
            sg.Checkbox("angry", key="angry", default=True, enable_events=True),
            sg.Checkbox("happy", key="happy", enable_events=True),
            sg.Checkbox("neutral", key="neutral", default=True, enable_events=True),
            sg.Checkbox("ps", key="ps", enable_events=True),
            sg.Checkbox("sad", key="sad", default=True, enable_events=True),
        ],
        # [sg.Button("OK1"), sg.Button("Cancel")],
    ]
    f_config_layout = [
        [sg.Text("请选择一个或多个特征：")],
        [
            sg.Checkbox("MFCC", key="mfcc", default=True, enable_events=True),
            sg.Checkbox("Mel", key="mel", enable_events=True),
            sg.Checkbox("Contrast", key="contrast", enable_events=True),
        ],
        [
            sg.Checkbox("Chromagram", key="chroma", enable_events=True),
            sg.Checkbox("Tonnetz", key="tonnetz", enable_events=True),
        ],
        # [sg.Button("OK2"), sg.Button("Cancel")],
    ]
    # ---column right---
    algos = get_algos_elements_list()
    len_of_algos = len(algos)
    algos_layout = [
        [sg.Text("选择一个算法进行试验:")],
        algos[: len_of_algos // 2],
        algos[len_of_algos // 2 :],
        # [sg.Button("OK3"), sg.Button("Cancel")],
    ]

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
        [
            sg.Button("OK", bind_return_key=True, key="file_choose_ok"),
            sg.Button("Cancel"),
        ],
    ]
    re_result = "暂无结果"
    emotion_recognition_layout = [
        [sg.Text("识别该语音文件的情感")],
        [sg.B("OK", key="start_recognize_emotion")],
        # [sg.Text(f"识别结果:{re_result}", key="emotion_recognition_res")],
        [create_border_frame(result=re_result, key="emotion_recognition_res")],
    ]

    draw_layout = [
        [sg.Text("请选择一个文件,绘制其[波形图|频谱图|Mel频谱图]：")],
        # [sg.Input(), sg.FileBrowse()],
        [
            sg.Checkbox("waveForm", key="wave_form"),
            sg.Checkbox("FreqGraph", key="freq_graph"),
            sg.Checkbox("MelFreqGraph", key="mel_freq_graph"),
        ],
        [sg.Button("OK", key="draw_graph"), sg.Button("Cancel")],
    ]
    # ---辅助信息---
    ML_KEY = "-ML-"  # Multline's key
    __version__ = "1.1"

    right_info_layout = [
        [
            sg.Multiline(
                size=(70, 21),
                write_only=True,
                # expand_x=True,
                expand_y=True,
                key=ML_KEY,
                reroute_stdout=True,
                echo_stdout_stderr=True,
                reroute_cprint=True,
            )
        ],
        [sg.T("CCSER Client By Cxxu_zjgsu " + __version__)],
        [
            sg.T(
                "PySimpleGUI ver "
                + sg.version.split(" ")[0]
                + "  tkinter ver "
                + sg.tclversion_detailed,
                font="Default 8",
                pad=(0, 0),
            )
        ],
        [sg.T("Python ver " + sys.version, font="Default 8", pad=(0, 0))],
        [
            sg.T(
                "Interpreter " + sg.execute_py_get_interpreter(),
                font="Default 8",
                pad=(0, 0),
            )
        ],
    ]
    # output tab
    logging_layout = [
        [sg.Text("Anything printed will display here!")],
        [
            sg.Multiline(
                size=(60, 15),
                font="Courier 8",
                # expand_x=True,
                # expand_y=True,
                write_only=True,
                reroute_stdout=True,
                reroute_stderr=True,
                echo_stdout_stderr=True,
                autoscroll=True,
                auto_refresh=True,
            )
        ],
    ]
    settings_layout = [
        [sg.Text("Settings")],
    ] + theme_layout
    # ---column divide---
    column_left = db_choose_layout + e_config_layout + f_config_layout
    column_right = (
        algos_layout
        + file_choose_layout
        + emotion_recognition_layout
        + draw_layout
        + right_info_layout
    )
    left_column_container = sg.Column(
        column_left, expand_x=True, expand_y=True, element_justification="l"
    )
    # column_middle_separator = sg.Column([[sg.VerticalSeparator()]], background_color='yellow')

    right_column_container = sg.Column(
        column_right, expand_x=True, expand_y=True, element_justification="c"
    )

    main_pane = sg.Pane(
        [
            left_column_container,
            #  [sg.VerticalSeparator(pad=None)],
            # column_middle_separator,
            right_column_container,
        ],
        orientation="h",
        expand_x=True,
        expand_y=True,
        k="-PANE-",
    )

    main_tab_layout = [
        [
            sg.Text(
                "Welcome to experience CCSER Client!",
                size=(45, 1),
                justification="center",
                font=("Helvetica", 16),
                relief=sg.RELIEF_RIDGE,
                k="-TEXT HEADING-",
                enable_events=True,
            )
        ],
        [main_pane],
        [sg.B("Quit")],
    ]
    main_page_layout = main_tab_layout

    # ----full layout----
    # --top Menu bar---
    Menubar_layout = [
        [sg.MenubarCustom(menu_def, key="-MENU-", font="Courier 15", tearoff=True)]
    ]

    # ---tabs---
    tabs_layout = [
        [
            sg.TabGroup(
                [
                    [
                        sg.Tab("Input Elements", main_tab_layout),
                        sg.Tab("Output", logging_layout),
                        sg.Tab("Settings", settings_layout),
                    ]
                ],
                key="-TAB GROUP-",
                expand_x=True,
                expand_y=True,
            ),
        ]
    ]

    layout = Menubar_layout + tabs_layout

    # ---create window---
    window = sg.Window(
        title="ccser_client",
        layout=layout,
        alpha_channel=0.9,
        resizable=True,
        size=size,
    )
    return window


audio_selected = ""

def initial(values):
    """收集组件的默认值,在用户操作前就应该扫描一遍设置在组件的默认值

    Parameters
    ----------

    values : dict
        当前系统的values值
    """
    train_db = values["train_db"]
    test_db = values["test_db"]
    e_config = scan_choosed_options(values)
    algorithm = selected_algo(values)
    f_config = selected_features(values)
    return train_db, test_db, e_config, algorithm, f_config



def main():
    window = make_window(size=size)
    # Initialize the selected databases list
    e_config = []
    f_config = []
    train_db = ""
    test_db = ""
    algorithm = ""

    while True:
        # 语料库的选择
        event, values = window.read()
        #初始化!
        train_db,test_db,e_config,algorithm,f_config = initial(values)

        if event in (None, "Exit"):
            print("Clicked Exit!")
            break

        elif event == "train_db":
            train_db = values["train_db"]
            print(train_db, "@{trian_db}")
        elif event == "test_db":
            test_db = values["test_db"]
            print(test_db, "@{test_db}")

        # ---情感组合的选择和下面的特征组合的选择逻辑一致,可以抽出相应逻辑复用
        # 这里采用两种不同的算法处理
        # 情感组合选择

        elif event in ava_emotions:
            e_config = scan_choosed_options(values)
            print(e_config, "@{e_config}")
        # 特征组合选择

        elif event in ava_features:
            # 遍历所有选项,检查对应的值是否为True
            # 一个思路是,这里我们只需要用户操作完后的这几个checkbox的状态(或者说哪些是True即可)
            # 可以每次操作这些checkbox中一个的时候,再扫描更新以下这些选项的信息即可
            f_config = selected_features(values)

            print(f_config, "@{f_config}")


        elif event in ava_algorithms:
            algorithm = selected_algo(values)

            print(algorithm, "@{algorithm}")
            # print(event, "处于选择algorithm的循环中.")
            # print("完成算法的选择.")

        # 这部分只负责选取文件,选取通过点击确认,来完成这部分逻辑,跳到循环,执行下一步分代码

        elif event == "file_choose_ok":
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
            global audio_selected
            audio_selected = values["-FILENAME-"]

            print(event, values["-FILENAME-"])

        elif event == "Clear History":
            sg.user_settings_set_entry("-filenames-", [])
            sg.user_settings_set_entry("-last filename-", "")
            window["-FILENAME-"].update(values=[], value="")
        # print("完成文件选取")
        # --情感识别阶段--
        elif event == "start_recognize_emotion":
            recognize(
                window, train_db, test_db, e_config, f_config, algorithm, audio_selected
            )

        elif event == "start_recognize_emotion":
            wave_form = values["wave_form"]
            freq_graph = values["freq_graph"]
            mel_freq_graph = values["mel_freq_graph"]
            # print(f"{event=}in draw tasks..(开始绘制.)")
        elif event == "start_recognize_emotion":
            if wave_form:
                showWaveForm(audio_selected)
            if freq_graph:
                showFreqGraph(audio_selected)
            if mel_freq_graph:
                showMelFreqGraph(audio_selected)

            # print("完成图形绘制.")

    print("关闭窗口.")

    window.close()

def selected_features(values):
    tmp_f_config = []
    for f in ava_features:
        used_feature = values[f]
        if used_feature:
                    # 使用插入的方式不是那么好,如果不设置一个临时变量来收集容易因为反复选取/撤销导致多余的选项出现
            tmp_f_config.append(f)
            # 扫描完毕,将结果更新为f_config的值
    f_config = tmp_f_config
    return f_config

def selected_algo(values):
    for algo in ava_algorithms:
        if values and values[algo]:
                    # 获取选中的算法名称(key)
            algorithm = algo
            break
    return algorithm

def scan_choosed_options(values):
    e_config_dict = dict(
                angry=values["angry"],
                happy=values["happy"],
                neutral=values["neutral"],
                ps=values["ps"],
                sad=values["sad"],
            )
    e_config = get_used_keys(e_config_dict)
    return e_config


def recognize(window, train_db, test_db, e_config, f_config, algorithm, audio_selected):
    print("开始识别..")
    print("检查参数..",)

    print('train_db:', train_db)
    print('test_db:', test_db)
    print('e_config:', e_config)
    print('f_config:', f_config)
    print('algorithm:', algorithm)
    print('audio_selected:', audio_selected)

    bclf_estimators = load(bclf)

    # audio_selected=get_example_audio_file()
    # None表示自动计算best_ML_model
    ML_estimators = {
        estimator.__class__.__name__: estimator for estimator, _, _ in bclf_estimators
    }
    ML_estimators["BEST_ML_MODEL"] = None
    # if algorithm=='BEST_ML_MODEL':
    model = ML_estimators[algorithm]
    print(train_db, test_db, e_config, f_config, algorithm, model, audio_selected)

    if algorithm == "RNN":
        der = DeepEmotionRecognizer(
            train_dbs=train_db, test_dbs=test_db, e_config=e_config, f_config=f_config
        )
        er = der
    else:
        er = EmotionRecognizer(
            model=model,
            train_dbs=train_db,
            test_dbs=test_db,
            e_config=e_config,
            f_config=f_config,
        )
    # 对数据进行训练(train方法自动导入数据)
    er.train()
    test_score = er.test_score()
    train_score = er.train_score()

    print(f"{test_score=}")
    print(f"{train_score=}")
    re_result = er.predict(audio_selected)
    print(f"{re_result=}")
    window["emotion_recognition_res"].update(f"{re_result}")


if __name__ == "__main__":
    main()
