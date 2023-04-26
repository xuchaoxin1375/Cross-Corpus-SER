##
import os
import inspect
import constants.beauty as bt
import data_visualization as dv
import fviewer
import ipdb
import PySimpleGUI as sg
import query as q
import constants.uiconfig as ufg
import constants.beauty as bt
from constants.beauty import (
    ccser_theme,
    db_introduction,
    h2,
    logo,
    option_frame,
    result_frame,
)
from demo_programs.Demo_Nice_Buttons import image_file_to_bytes, red_pill64, wcolor
from fviewer import audio_viewer_layout, fviewer_events, selected_files
from joblib import load
from multilanguage import get_your_language_translator
from constants.uiconfig import ML_KEY, __version__
from user import UserAuthenticatorGUI


lang = get_your_language_translator("English")
import sys

# from psgdemos import *
# from SG.psgdemos import find_in_file, get_editor, get_explorer, get_file_list, filter_tooltip, find_re_tooltip, find_tooltip, get_file_list_dict, settings_window, using_local_editor, window_choose_line_to_edit
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
    speech_dbs_dir,
)


def import_config_bookmark():
    pass


def define_constants():
    pass


## constants


size = (1500, 1000)
# size=None

train = "trian"
test = "test"
algorithm = ""
audio_selected = ""
speech_folder = speech_dbs_dir
no_result_yet = f"No Result Yet"
predict_res_key = "emotion_predict_res"
predict_proba_tips_key = "predict_proba"
train_result_table_key = "train_result_table"
predict_proba_table_key = "predict_proba_table"
predict_proba_frame_key = "predict_proba_frame"
predict_proba_table_frame_key = "predict_proba_table_frame"
userUI = UserAuthenticatorGUI()
# ---辅助信息---


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


##
# ---create the window---
def make_window(theme=None, size=None):
    if theme:
        # print(theme)
        sg.theme(theme)
    menu_def = [["&Application", ["E&xit"]], ["Help", ["Introduction"]]]
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
                size=bt.lb_size,
                key="-THEME LISTBOX-",
                enable_events=True,
            )
        ],
        [sg.Button("Set Theme")],
    ]
    # ---file viewer--
    # file_viewer_layout = file_view_layout()
    # ---create 2 column layout---
    # ---column left---
    db_choose_layout = [
        [bt.h2("Select the training database")],
        [sg.Combo(ava_dbs, key="train_db", default_value=emodb, enable_events=True)],
        [bt.h2("Select the testing database")],
        [sg.Combo(ava_dbs, key="test_db", default_value=emodb, enable_events=True)],
    ]  # shape=(-1,1)

    # [sg.Checkbox(emo) for emo in ava_emotions]
    emotion_config_checboxes_layout = [
        [
            sg.Checkbox("angry", key="angry", default=True, enable_events=True),
            sg.Checkbox("happy", key="happy", enable_events=True),
            sg.Checkbox("neutral", key="neutral", default=True, enable_events=True),
            sg.Checkbox("ps", key="ps", enable_events=True),
            sg.Checkbox("sad", key="sad", default=True, enable_events=True),
            sg.Checkbox("others", key="others", default=True, enable_events=True),
        ]
    ]

    e_config_layout = [
        [
            bt.h2(
                text="choose the emotion config",
                # relief=sg.RELIEF_SOLID,
                # style_add='underline',
                style_add="italic",
                tooltip=lang["choose_emotion_config"],
            ),
        ],
        [
            bt.option_frame(
                title="Emotion Config chooser", layout=emotion_config_checboxes_layout
            )
        ],
    ]
    f_config_option_frame = option_frame(
        title="Feature Config chooser",
        layout=[
            [
                sg.Checkbox("MFCC", key="mfcc", default=True, enable_events=True),
                sg.Checkbox("Mel", key="mel", enable_events=True),
                sg.Checkbox("Contrast", key="contrast", enable_events=True),
                # 可以考虑在这里换行
                # ],
                # [
                sg.Checkbox("Chromagram", key="chroma", enable_events=True),
                sg.Checkbox("Tonnetz", key="tonnetz", enable_events=True),
            ],
        ],
        frame_key="f_config_layout",
    )
    f_config_layout = [
        [bt.h2(lang["choose_feature_config"])],
        [f_config_option_frame],
    ]
    # ---column right---
    algos = get_algos_elements_list()
    len_of_algos = len(algos)

    algo_frame = option_frame(
        title="Algorithms chooser",
        layout=[
            algos[: len_of_algos // 2],
            algos[len_of_algos // 2 :],
        ],
        frame_key="algo_border_frame",
    )
    algos_layout = [
        [bt.h2(lang["choose_algorithm"])],
        [algo_frame],
    ]

    file_choose_layout = [
        [bt.h2(lang["choose_audio"])],
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
            # sg.Button("Cancel"),
            # ],
            # [
            sg.B(
                "recognize it",
                key="recognize it",
                tooltip=lang["recognize_the_audio_emotion"],
            ),
        ],
    ]
    train_result_table_layout = [
        [
            sg.Table(
                values=[["pending"] * 2],
                headings=["train_score", "test_score"],
                justification="center",
                font="Arial 16",
                expand_x=True,
                key=train_result_table_key,
                num_rows=1,  # 默认表格会有一定的高度,这里设置为1,避免出现空白
                hide_vertical_scroll=True,
            )
        ]
    ]
    train_fit_button_layout = [
        [
            # sg.Button('start train'),
            sg.RButton(
                "start train",
                image_data=image_file_to_bytes(red_pill64, (100, 50)),
                button_color=("white", "white"),
                # button_color=wcolor,
                font="Any 15",
                pad=(0, 0),
                key="start train",
            ),
        ]
    ]
    train_result_layout = [
        [
            bt.result_frame(
                title=lang["train_result_title"],
                layout=train_result_table_layout,
                frame_key="train_result_frame",
            ),
        ]
    ]
    # bt.res_content_layout(no_result_yet,res_key='train_result')
    predict_res_layout = bt.res_content_layout(
        text=no_result_yet, justification="c", key=predict_res_key
    )
    # predict_proba_tips_layout = [[sg.Text("pending", key=predict_proba_tips_key)]]
    #默认不显示predict_proba的不可用说明
    predict_proba_tips_layout = bt.normal_content_layout(
        text="pending", key=predict_proba_tips_key,visible=False
    )
    #默认显示predict_proba表格
    predict_proba_table_layout = [
        [
            sg.Table(
                values=[["pending"] * 2],
                headings=["emotino", "proba"],
                justification="c",
                font="Arial 16",
                expand_x=True,
                expand_y=False,
                key=predict_proba_table_key,
                auto_size_columns=True,
                display_row_numbers=True,
                num_rows=1,
            )
        ]
    ]
    predict_prob_layout=[
        *predict_proba_tips_layout,
        *predict_proba_table_layout
    ]
    predict_res_frames_layout = [
        [result_frame(layout=predict_res_layout)],
        [
            result_frame(
                title="predict_proba_tips",
                layout=predict_prob_layout,
                frame_key=predict_proba_frame_key,
                # visible=False,
            )
        ],
        # [
        #     result_frame(
        #         title="predict_proba_table",
        #         layout=predict_proba_table_layout,
        #         frame_key=predict_proba_table_frame_key,
        #     ),
        # ],
    ]
    draw_layout = [
        [bt.h2(lang["draw_diagram"], tooltip=lang["draw_diagram_detail"])],
        # [sg.Input(), sg.FileBrowse()],
        [
            sg.Checkbox("waveForm", key="wave_form"),
            sg.Checkbox("FreqGraph", key="freq_graph"),
            sg.Checkbox("MelFreqGraph", key="mel_freq_graph"),
        ],
        # todo reset
        [sg.Button("draw_graph"), sg.Button("Reset", key="reset graph Checkbox")],
    ]

    info_layout = [
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
    # dbs_introduce_layout=[
    #     [sg.Text("数据库选择")],
    # ]
    # output tab
    analyzer_layout = (
        [
            [bt.h2("Anything printed will display here!")],
            [
                sg.Multiline(
                    size=bt.ml_size,
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
        + dv.layout
        + q.query_layout
    )

    settings_layout = [
        [sg.Text("Settings")],
    ] + theme_layout
    about_layout = info_layout
    # ---column left---

    left_col_layout = (
        db_choose_layout
        + e_config_layout
        + f_config_layout
        + algos_layout
        + train_fit_button_layout
        + train_result_layout
        + file_choose_layout
        + predict_res_frames_layout
        + draw_layout
        # + file_viewer_layout
    )
    right_column_layout = (
        #  [
        #     [
        #         sg.Button("open folder"),
        #         sg.Text("<folder of speech db>", key="speech_folder_path"),
        #     ],
        # ]
        # +
        audio_viewer_layout
        + [
            [sg.Text("dev logging tool:")],
            [sg.HorizontalSeparator(color=bt.seperator_color)],
            [
                sg.Multiline(
                    size=bt.ml_size,
                    write_only=True,
                    # expand_x=True,
                    expand_y=True,
                    key=ML_KEY,
                    reroute_stdout=True,
                    echo_stdout_stderr=True,
                    reroute_cprint=True,
                    auto_refresh=True,
                    autoscroll=True,
                )
            ],
        ]
    )
    left_column = sg.Column(
        left_col_layout,
        expand_x=True,
        expand_y=True,
        element_justification="l",
        scrollable=True,
        vertical_scroll_only=True,
    )
    # column_middle_separator = sg.Column([[sg.VerticalSeparator()]], background_color='yellow')

    right_column = sg.Column(
        right_column_layout,
        expand_x=True,
        expand_y=True,
        element_justification="l",
        scrollable=True,
        vertical_scroll_only=True,
    )

    main_pane = sg.Pane(
        [
            left_column,
            #  [sg.VerticalSeparator(pad=None)],
            # column_middle_separator,
            right_column,
        ],
        orientation="h",
        expand_x=True,
        expand_y=True,
        k="-PANE-",
    )
    main_pane_layout = [[left_column, right_column]]
    global userUI
    userUI = UserAuthenticatorGUI()
    user_layout = [
        # [sg.Text("Welcome:"),sg.Text("User",key=current_user_key)],
        # [sg.Input(default_text="user name or ID",key="-USER-")],
        # [sg.Input(default_text="password",key="-PASSWORD-")],
    ] + userUI.create_user_layout()

    main_tab_layout = [
        [
            sg.Text(
                # "Welcome to experience CCSER Client!",
                lang["welcome_title"],
                size=bt.welcom_title_size,
                justification="center",
                font=("Comic", 50),
                relief=sg.RELIEF_RIDGE,
                k="-TEXT HEADING-",
                enable_events=True,
                expand_x=True,
            )
        ],
        # [main_pane],
        # [sg.B(ufg.close)],
    ] + main_pane_layout
    # main_page_layout = main_tab_layout

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
                        sg.Tab("WelcomeUser", user_layout),
                        sg.Tab("MainPage", main_tab_layout),
                        sg.Tab("Analyzer", analyzer_layout),
                        sg.Tab("Settings", settings_layout),
                        sg.Tab("about", about_layout),
                    ]
                ],
                key="-TAB GROUP-",
                expand_x=True,
                expand_y=True,
            ),
        ]
    ]

    layout = Menubar_layout + tabs_layout
    # layout=theme_layout

    # ---create window---
    window = sg.Window(
        title="ccser_client",
        layout=layout,
        # alpha_channel=0.9,
        resizable=True,
        size=size,
    )
    return window


def initial(values=None, verbose=1):
    """收集组件的默认值,在用户操作前就应该扫描一遍设置在组件的默认值

    Parameters
    ----------

    values : dict
        当前系统的values值
    """
    # event, values = window.read()
    train_db = values["train_db"]
    test_db = values["test_db"]
    e_config = scan_choosed_options(values)
    algorithm = selected_algo(values)
    f_config = selected_features(values)
    if verbose >= 2:
        # print(train_db, test_db, e_config, algorithm, f_config)
        print(f"train_db = {train_db}")
        print(f"test_db = {test_db}")
        print(f"e_config = {e_config}")
        print(f"algorithm = {algorithm}")
        print(f"f_config = {f_config}")
    return train_db, test_db, e_config, algorithm, f_config


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
    global algorithm
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
        others=values["others"],
    )
    e_config = get_used_keys(e_config_dict)
    return e_config


def proba_available(er):
    """
    the function is a inner_function of recognize_audio
    This function checks if the classifier supports probability estimates.

    params
    -
    :param er: The emotion recognition model object.
    :return: True if the classifier supports probability estimates, False otherwise.
    """
    model = er.model
    res = hasattr(model, "predict_proba")
    if res:
        print("Classifier supports probability estimates")
    else:
        print("Classifier does not support probability estimates")
    return res


def recognize_auido(
    window=None,
    er=None,
    # train_db=None,
    # test_db=None,
    # e_config=None,
    # f_config=None,
    # algorithm=None,
    audio_selected=None,
):
    """
    This function performs audio recognition and updates the GUI window with the result.

    params
    -
    :param window: The GUI window object.
    :param train_db: The training database.
    :param test_db: The testing database.
    :param e_config: The configuration file for the emotion recognition model.
    :param f_config: The configuration file for the feature extraction model.
    :param algorithm: The algorithm to be used for emotion recognition.
    :param audio_selected: The selected audio file for recognition.
    :return: None
    """
    print("audio_selected:", audio_selected)
    if not audio_selected:
        # audio_selected = get_example_audio_file()
        sys.exit("Please select an audio file at first!")
    if er is None:
        sg.popup("Please train an emotion recognition model at frist !")
        # return
        # er = start_train_model(
        #     train_db=train_db,
        #     test_db=test_db,
        #     e_config=e_config,
        #     f_config=f_config,
        #     algorithm=algorithm,
        # )
    else:
        emotion_predict_result = er.predict(audio_selected)
        print(f"{emotion_predict_result=}")
        # 更新结果
        window[predict_res_key].update(f"{emotion_predict_result}")
    # --处理置信度--
    if proba_available(er):
        predict_proba = er.predict_proba(audio_selected)

        # window[predict_proba_res_key].update(f"{predict_proba}")

        data = list(predict_proba.items())
        # print(data,"@{data}")
        data = [[emo, proba] for emo, proba in data]
        #关闭proba_tip的显示
        window[predict_proba_tips_key].update(visible=False)
        #更新proba表格内容
        # window[predict_proba_tips_frame_key].update(visible=False)
        ppt = window[predict_proba_table_key]
        # inspect.getfullargspec(ppt.update)
        ppt.update(
            values=data,
            num_rows=4,
            #    display_row_numbers=True
            visible=True
        )
        # window[]
        # window[predict_proba_table_frame_key].update(visible=True)

    else:
        window[predict_proba_tips_key].update(
            value=(
                "The parameter setting of this model is to disable confidence calculation,\nif you'd like to view predict_proba,try another model like RF"
            ),
            visible=True,
        )
        #关闭表格的显示
        window[predict_proba_table_key].update(visible=False)
    window.refresh()


def start_train_model(
    train_db=None, test_db=None, e_config=None, f_config=None, algorithm=None, verbose=1
):
    """
    Train an emotion recognition model and returns an EmotionRecognizer object.

    Args:
        train_db (list): List of training audio file paths.
        test_db (list): List of testing audio file paths.
        e_config (dict): Configuration dictionary for audio feature extraction.
        f_config (dict): Configuration dictionary for audio feature selection.
        algorithm (str): Name of the machine learning algorithm to use for training.
        verbose (int): Level of verbosity. 0 for no output, 1 for standard output.

    Returns:
        er (EmotionRecognizer): Trained emotion recognition model.
    """
    print("start to train the model ..")
    print(
        "checking arguments..",
    )
    from recognizer.basic import EmotionRecognizer

    if verbose:
        print("train_db:", train_db)
        print("test_db:", test_db)
        print("e_config:", e_config)
        print("f_config:", f_config)
        print("algorithm:", algorithm)

    bclf_estimators = load(bclf)

    # audio_selected=get_example_audio_file()
    # None表示自动计算best_ML_model
    ML_estimators_dict = {
        estimator.__class__.__name__: estimator for estimator, _, _ in bclf_estimators
    }
    # ipdb.set_trace()
    if isinstance(algorithm, list):
        sys.exit()
    ML_estimators_dict["BEST_ML_MODEL"] = None
    # if algorithm=='BEST_ML_MODEL':
    model = ML_estimators_dict[algorithm]
    print(train_db, test_db, e_config, f_config, algorithm, model, audio_selected)

    if algorithm == "RNN":
        from recognizer.deep import DeepEmotionRecognizer

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
            verbose=1,
        )
    # 对数据进行训练(train方法自动导入数据)
    er.train()
    # model_res(er,verbose=verbose)
    return er


def model_res(er, verbose=1):
    """
    Computes the train and test scores of a given model.

    Args:
        er (estimator): A trained estimator object.
        verbose (int): Whether or not to print the test and train scores.

    Returns:
        tuple: A tuple containing the train score and test score.
    """
    train_score = er.train_score()
    test_score = er.test_score()
    if verbose:
        print(f"{er.model=}")
        print(f"{test_score=}")
        print(f"{train_score=}")
    return train_score, test_score


##
def main(verbose=1):
    theme = ccser_theme
    window = make_window(theme=theme, size=size)
    # Initialize the selected databases list
    event, values = window.read()
    e_config = []
    f_config = []
    train_db = ""
    test_db = ""
    algorithm = ""
    er = None
    # 初始化!
    train_db, test_db, e_config, algorithm, f_config = initial(values=values, verbose=2)

    while True:
        if verbose >= 2:
            check_training_arguments(e_config, f_config, train_db, test_db, algorithm)

        if event:  # 监听任何event
            print(event, "@{event}", __file__)

        # 语料库的选择
        if event in (ufg.close, sg.WIN_CLOSED):
            print(ufg.close, "关闭窗口")
            break
        elif event == "train_db":
            train_db = values["train_db"]
            if verbose > 1:
                print(train_db, "@{trian_db}")
        elif event == "test_db":
            test_db = values["test_db"]
            if verbose > 1:
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
            if verbose > 1:
                print(f_config, "@{f_config}")

        elif event in ava_algorithms:
            algorithm = selected_algo(values)
            if verbose:
                print(algorithm, "@{algorithm}")
                # print(event, "处于选择algorithm的循环中.")
                # print("完成算法的选择.")

        # 这部分只负责选取文件,选取通过点击确认,来完成这部分逻辑,跳到循环,执行下一步分代码

        elif event == "file_choose_ok":
            # If OK, then need to add the filename to the list of files and also set as the last used filename
            file_selected_record(verbose, event, values)

        elif event == "Clear History":
            clear_history(window)
        # ---文件夹选取---
        elif event == "open folder":
            open_folder_event(window)
            # print("完成文件选取")
        # --情感识别阶段--
        elif event == "start train":
            er = start_train_model(
                train_db=train_db,
                test_db=test_db,
                e_config=e_config,
                f_config=f_config,
                algorithm=algorithm,
            )
            # 训练收尾工作:将计算结果(识别器)传递给fviewer,赋能fviewer可以(直接利用识别器对象)进行识别

            fviewer.er = er  # 是否为多余#TODO
            train_score, test_score = model_res(er, verbose=verbose)
            # window["train_result"].update(f"{train_score=},{test_score=}")
            res = [round(x, 4) for x in (train_score, test_score)]
            window["train_result_table"].update(
                values=[res]
            )  # values类型是list[list[any]],每个内部列表表示表格的一个行的数据

        elif event == "recognize it":
            recognize_auido(
                window=window,
                er=er,
                # train_db=train_db,
                # test_db=test_db,
                # e_config=e_config,
                # f_config=f_config,
                # algorithm=algorithm,
                audio_selected=audio_selected,
            )

        elif event == "draw_graph":
            wave_form = values["wave_form"]
            freq_graph = values["freq_graph"]
            mel_freq_graph = values["mel_freq_graph"]
            # print(f"{event=}in draw tasks..(开始绘制.)")
            if wave_form:
                showWaveForm(audio_selected)
            if freq_graph:
                showFreqGraph(audio_selected)
            if mel_freq_graph:
                showMelFreqGraph(audio_selected)
            # print("完成图形绘制.")

        elif event == "Set Theme":
            # print("[LOG] Clicked Set Theme!")
            select_items_list = values["-THEME LISTBOX-"]
            theme_chosen = values["-THEME LISTBOX-"][0]
            if verbose:
                print(select_items_list, "@{select_item}")
                print("[LOG] User Chose Theme: " + str(theme_chosen))
            window.close()
            print("the window was closed!")
            # window = make_window(theme=theme_chosen)
            # if(window):
            #     print("restart successful!")
            # window = make_window()
        elif event == "Introduction":
            content = [logo, db_introduction]
            res = "\n".join(content)
            sg.popup_scrolled(res, size=(150, 100), title="Introduction")
        else:
            # 具有独立的事件循环,直接调用即可
            userUI.run_module(event, values, window=window, verbose=1)
            q.query_events(event, values, theme=theme)
        #!如果希望每轮循环都要运行的代码就从if/elif断开,写在这里
        # audio_viewer事件循环模块
        fviewer_events(window, event, values)

        #!请在上面添加事件循环
        # 本例在事件循环之前已经调用过一次read()方法,如果连续两次调用中间没有没有对事件进行捕获,那么第一次的事件将会丢失
        event, values = window.read()

    print("关闭窗口.")

    window.close()


def open_folder_event(window):
    print("[LOG] Clicked Open Folder!")
    folder_or_file = sg.popup_get_folder(
        "Choose your folder", keep_on_top=True, default_path=speech_dbs_dir
    )

    speech_folder_path = str(folder_or_file)
    sg.popup("You chose: " + speech_folder_path, keep_on_top=True)
    print("[LOG] User chose folder: " + speech_folder_path)
    window["speech_folder_path"].update(speech_folder_path)


def clear_history(window):
    sg.user_settings_set_entry("-filenames-", [])
    sg.user_settings_set_entry("-last filename-", "")
    window["-FILENAME-"].update(values=[], value="")


def file_selected_record(verbose, event, values):
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
    if verbose:
        print(event, values["-FILENAME-"])


def check_training_arguments(e_config, f_config, train_db, test_db, algorithm):
    print(f"train_db = {train_db}")
    print(f"test_db = {test_db}")
    print(f"e_config = {e_config}")
    print(f"algorithm = {algorithm}")
    print(f"f_config = {f_config}")


if __name__ == "__main__":
    main()

##
