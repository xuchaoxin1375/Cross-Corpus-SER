##
import os
import time
from pathlib import Path
import PySimpleGUI as sg
import re
from config.MetaPath import speech_dbs_dir, savee
import constants.uiconfig as ufg
import table_show as ts
import constants.beauty as bt
from SG.multilanguage import get_language_translator

# from recognizer.basic import EmotionRecognizer
import data_visualization as dv

# from SG.translations import en,zh
language = "zh"
lang = get_language_translator(language)
dv.lang=lang
ts.lang=lang
# 主题设置说明:当主题设置语句防止在程序的末尾时可能是无效的
# 猜测sg.theme()设置完主题后,后续在调用sg的元素创建方法才会有相应主题的配色
# 如果控件都已经创建好了才开始调用sg.theme()修改配色,那来不及起作用了
sg.theme(bt.ccser_theme)
# 常量


# 将变量设置在事件循环中可能会反复被初始化,这里我们应该放在事件循环的外部
# speech_folder_path=Path("./")
speech_folder = speech_dbs_dir
emodb_dir = speech_dbs_dir / "emodb/wav"
audio_exts = (".wav", ".mp3", ".ogg")
savee_dir = speech_dbs_dir / "savee/AudioData"

selected_files = []
# er: EmotionRecognizer = None
er = None
emotion_count_ts: ts.TableShow = None

selected_files_tooltip = lang.selected_files_tooltip

# filter_tooltip =

filter_input_key = "filter_input"
files_browsed_key = "files browsed"
selected_files_listbox_key = "selected_files_list"
num_selected_files_key = "num_selected_files"
num_files_key = "num_files_text"
short_path_checkbox_key = "short_path"
recursive_checkbox_key = "recursive_checkbox"
auto_refresh_checkbox_key = "auto_refresh"
audio_file_list_key = "audio_files_list"
confirm_files_selected_key = "confirm files selected"
confirm_folder_selected_key = "confirm folder selected"
filter_audios_key = "filter audios"
speech_folder_path_input_key = "speech_folder_path_input"
speech_folder_path_chooser_key = "speech_folder_path_chooser"


##
def get_audios_regex(
    recursive=False,
    speech_folder_root=speech_folder,
    filter_regex="",
    short=True,
    verbose=1,
):
    """
    Get a list of audio files from a given folder path.

    Args:
        recursive (bool): Whether to search subdirectories recursively. Default is False.
        speech_folder_root (str): Path to the folder where audio files are located.
        filter_regex (str): Regular expression to filter files by name. Default is empty.

    Returns:
        list: A list of audio file paths.
    """
    audios = []
    p = Path(speech_folder_root)
    if recursive:
        audios = get_audios(speech_folder_root, audio_exts, recursive=True)

    else:
        audios = get_audios(
            speech_folder_root, audio_exts, pattern="*", recursive=False
        )
    # 调试模式,切片出一小部分来试验算法
    # audios=audios[:50]

    # 正则表达式模式
    if filter_regex:
        if verbose:
            print("filter_regex:> ", filter_regex)
        # 由于这里需要反复使用正则匹配,因此采用编译的方式来提高性能
        p = re.compile(filter_regex, re.IGNORECASE)
        # s = re.search(filter_regex, str(path), re.IGNORECASE)

    filtered_audios = []
    for path in audios:
        # print(path)
        # 处理路径长短模式
        if short:
            # 对路做压缩处理(不显示speech_folder_root这部分)
            path = path.relative_to(speech_folder_root)
        else:
            path = path.absolute()
        # 对路径进行正则过滤
        # todo 对括号的识别有问题(得益于模块化,可以直接在这个模块内启动图形界面进行调试)
        if filter_regex:
            s = p.search(str(path))
            if s:
                filtered_audios.append(path)
        else:
            filtered_audios.append(path)

    return filtered_audios


##
def get_audios(folder, exts, pattern="*", recursive=False, flatten=True, verbose=0):
    """
    Find audio files in a folder with specific extensions.

    Arguments:
    - folder (pathlib.Path): the folder to search for audio files
    - exts (list of str): the extensions of the audio files to look for
    - pattern (str, optional): the pattern to match for the audio files (default "*")
    - flatten (bool, optional): whether to flatten the sub-lists of audio files (default True)
    - verbose (int, optional): whether and how much to print information about the search (default 1)

    Returns:
    - audio_files (list of pathlib.Path): the paths to the audio files found in the folder
    """
    if recursive:
        audio_files = [list(folder.rglob(f"{pattern}{ext}")) for ext in exts]
    else:
        audio_files = [list(folder.glob(f"{pattern}{ext}")) for ext in exts]

    if verbose:
        print({ext: len(audios) for audios, ext in zip(audio_files, exts)})
    # audio_files=[audio for audio in category for category in audio_files]
    audio_files_flatten = []
    if flatten:
        for category in audio_files:
            audio_files_flatten += category
        audio_files = audio_files_flatten
        # print(f"{len(audio_files_flatten)=}")
    return audio_files


##
# 创建GUI窗口
folder_browse_init_dir = speech_dbs_dir / savee  # 作为一个初始值

default_folder_file_list = get_audios_regex(
    recursive=True, speech_folder_root=speech_dbs_dir, short=True
)

# print(default_folder_file_list[:10])
##
len_default_folder_file_list = len(default_folder_file_list)

right_click_menu = [
    "",
    [lang.show_file_path, lang.show_file_size, lang.play_audio, lang.emotion_recognize],
]


files_selected_prompt = lang.files_selected_prompt


def audio_viewer_layout(theme=""):
    if theme:
        sg.theme(theme)
    audio_listbox_values = [
        lang.click_filter_prompt,
        lang.listbox_default_value_prompt,
    ]

    audio_viewer_layout = [
        [
            sg.Text(lang.select_dir_prompt),
            #  change the visible to True to try the language and theme switch!
            sg.Button("restart", visible=False),
        ],
        [
            sg.InputText(
                default_text=speech_folder,
                key=speech_folder_path_input_key,
                tooltip=lang.path_input_tooltip,
            ),
            sg.FolderBrowse(
                initial_folder=folder_browse_init_dir,
                button_text=lang.folder_browse,
                change_submits=True,
                key=speech_folder_path_chooser_key,
                target=speech_folder_path_input_key,
                # enable_events=True,
                tooltip=f"{lang.choose_folder_tooltip}{speech_folder}",
            ),
        ],
        [sg.Button(lang.confirm_folder_selected, key=confirm_folder_selected_key)],
        [
            sg.Input(
                default_text=files_selected_prompt,
                key=files_browsed_key,
            ),
            # sg.Text(text="files selected by filesBrowse will be shown \n in the listbox below"),
            sg.FilesBrowse(
                button_text=lang.files_browse,
                target=files_browsed_key,
                key="FilesBrowse",
                enable_events=True,
                change_submits=True,
            ),
        ],
        [
            sg.OK(
                button_text=lang.confirm_files_selected_button,
                key=confirm_files_selected_key,
            ),
        ],
        [
            sg.Text(lang.current_directory_prompt),
            sg.Text(f"{speech_folder}", key="current_dir"),
        ],
        [
            sg.Checkbox(
                text=lang.recursively_scan_subdir,
                default=True,
                key=recursive_checkbox_key,
                enable_events=True,
            ),
            sg.Checkbox(
                text=lang.auto_refresh,
                default=False,
                key=auto_refresh_checkbox_key,
                enable_events=True,
            ),
            sg.Checkbox(
                text=lang.short_path,
                default=True,
                key=short_path_checkbox_key,
                enable_events=True,
            ),
        ],
        [
            sg.Text(lang.filter_by_regex_prompt),
            sg.InputText(key="filter_input", default_text="", enable_events=True),
        ],
        [
            sg.Button(
                button_text=lang.filter_audios,
                key=filter_audios_key,
                tooltip=lang.auto_refresh_tooltip,
            ),
            # sg.Button(ufg.close),
        ],
        [sg.Text(f"{len_default_folder_file_list} {lang.files_count_unit}", key="num_files_text")],
        [
            sg.Listbox(
                values=default_folder_file_list,
                # size=(50, 10),
                size=bt.lb_size,
                key=audio_file_list_key,
                enable_events=True,
                bind_return_key=True,
                tooltip=lang.filter_tooltip,
                # 定义位于列表中条目的右键菜单内容
                right_click_menu=right_click_menu,
                select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED,
                no_scrollbar=True,
            )
        ],
        [
            sg.Text(lang.selected_audios_prompt),
            sg.Text(lang.no_files, key=num_selected_files_key),
        ],
        [
            sg.Listbox(
                values=audio_listbox_values,
                size=bt.lb_size,
                key=selected_files_listbox_key,
                tooltip=selected_files_tooltip,
                right_click_menu=right_click_menu,
                select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED,
            )
        ],
    ]
    return audio_viewer_layout


def make_window(theme):
    window = sg.Window(lang.audio_viewer, audio_viewer_layout(theme), resizable=True)
    return window


# 定义文件大小计算函数
def get_file_size(file_path):
    """
    Given a file path, returns a string representation of the file size in human-readable format.
    The size is rounded to two decimal places and the unit is chosen from a list of size names
    based on the size in bytes: "Bytes", "KB", "MB", "GB". If the file does not exist or is not
    accessible, an exception is raised.

    Args:
        file_path (str): The path to the file to check.

    Returns:
        str: A string representation of the file size, like "3.14 MB".

    Raises:
        OSError: If the file does not exist or is not accessible.
    """

    size = os.path.getsize(file_path)
    size_name = ["Bytes", "KB", "MB", "GB"]
    i = 0
    while size >= 1024 and i < len(size_name) - 1:
        size /= 1024.0
        i += 1
    return f"{size:.2f} {size_name[i]}"


# 事件循环
def get_absolute_path(speech_folder_path, selected_file, verbose=0):
    """
    Returns the absolute path of the selected file in the speech folder path.

    :param speech_folder_path: The path of the speech folder.
    :param selected_file: The name of the selected file.
    :param verbose: If True (1), print the audio path.
    :return: The absolute path of the selected file.
    """
    selected_file = Path(speech_folder_path) / selected_file
    audio_path = selected_file.absolute().as_posix()
    if verbose:
        print(audio_path, "@{audio_path}")
    # print(selected_file,"@{selected_file}")
    return audio_path


def get_abs_selected_pathes(speech_folder_path, selected_files):
    abs_pathes = []
    for selected_file in selected_files:
        # values["audio_files_list"]
        abs_path = get_absolute_path(speech_folder_path, selected_file)
        abs_pathes.append(abs_path)
    return abs_pathes


def main():
    layout = audio_viewer_layout()
    global lang
    window = sg.Window(lang.audios_filter, layout, resizable=True)
    while True:
        event, values = window.read()
        print(event, "@{event} main")
        if event in (sg.WINDOW_CLOSED, ufg.close):
            break
        elif event == "restart":
            window.close()
            print("closed successfully!")
            lang = get_language_translator("zh")
            window = make_window(theme="Reds")
        else:
            # 处理事件(小心,如果下面的函数编写不当,可能使得某些控件不能够正常工作)
            # 例如,FolderBrowser生成的按钮点击无法呼出系统的资源管理器(或者需要反复点击)
            pass
            fviewer_events(window, event, values)

    window.close()


def fviewer_events(window, event=None, values=None, verbose=1):
    global selected_files
    global speech_folder
    global emotion_count_ts
    # 处理 "filter_input" 事件
    if verbose:
        print("[Ev]", event, "@{event}", __file__)
    # refresh_viewer(window, values)
    need_update_list = (
        confirm_folder_selected_key,
        filter_input_key,
        filter_audios_key,
        short_path_checkbox_key,
        recursive_checkbox_key,
    )
    if event in need_update_list:
        # 判断手动输入的路径是否合法
        if event == confirm_folder_selected_key:
            path = values[speech_folder_path_input_key]
            print(path, "was confirmed!")
            if Path(path).exists():
                speech_folder = path
                # 更新当前speech_path控件
                window["current_dir"].Update(path)
                # 更新文件列表视图
                refresh_viewer(window, speech_folder=path, values=values)
            else:
                sg.popup_error(f"{path} {lang.not_exist}")
        # 刷新文件列表
        elif event == filter_input_key and not values[auto_refresh_checkbox_key]:
            return

        refresh_viewer(window, speech_folder=speech_folder, values=values)

    # elif event == speech_folder_path_chooser:
    #     print(f"you clicked folderbrowser: {speech_folder_path_chooser}")
    elif event == confirm_files_selected_key:
        # 处理多选文件按钮的返回结果
        selected_files = values[files_browsed_key].split(";")
        print(selected_files, "@{selected_files}")
        refresh_selected_view(window, len(selected_files))

    elif event == audio_file_list_key:
        # 处理 "audio_files_list" 事件
        selected_files = values[audio_file_list_key]
        num_selected_files = len(selected_files)
        # 更新选中列表视图控件

        refresh_selected_view(window, num_selected_files)

    # 处理 "Show File Path" 事件
    elif event == lang.show_file_path:
        res = []
        for file in selected_files:
            res.append(get_absolute_path(speech_folder, file))
        selected_file_pathes = "\n".join(res)
        sg.popup(selected_file_pathes, title="File Path")

        # 处理 "Show File Size" 事件
    elif event == lang.show_file_size:
        # selected_file = get_abs_selected_pathes(speech_folder_path, selected_files)
        res = []
        for selected_file in selected_files:
            selected_file = get_absolute_path(speech_folder, selected_file)
            file_size = os.path.getsize(selected_file)
            size_str = get_file_size(selected_file)
            sentence = f"The file <{selected_file}>size is {size_str}."
            res.append(sentence)
        res = "\n".join(res)
        sg.popup(f"{res}", title=lang.file_size)
        # 处理 "Play Audio" 事件
    elif event == lang.play_audio:
        pathes = get_abs_selected_pathes(speech_folder, selected_files)
        print(pathes, selected_files)

        from pydub import AudioSegment
        from pydub.playback import play

        for audio_path in pathes:
            # 读取音频文件
            name, ext = os.path.splitext(audio_path)
            print(name, "@{name}", ext, "@{ext}")
            # 播放音频
            audio_file = AudioSegment.from_file(audio_path, format=ext)
            play(audio_file)
    elif event == lang.emotion_recognize:
        # print()
        # 为了完成多选文件(成批识别),经过brainstorm,提出以下idea:
        # 委托给ccser_gui模块来处理,通过共享变量来简单通信/创建一个媒介模块来解决相互导入的问题(对于这种简单的场景够用的)
        # 如果出现两个模块相互导入,那么往往要考虑包相互导入的部分中哪些东西抽去到单独的模块中,优化模块的结构
        # 在ccser_gui模块中调用本模块的方法时,采用传参的方式是最直接的通信方式(只不过有些调用参数很多,需要传比较多的参数😂)
        # 幸运的是,在python中支持动态添加类(成员属性),可以通过将需要传递的值保留在类的实例中,这样可以减少调用时需要传递的参数(特别时反复用到相关数据时,这更有用)
        # 这里的识别应该在训练阶段完成之后才调用的,否则程序应该组织这样跨阶段的行为,提高robustness
        if er == None:
            print("请先完成识别器训练,然后再执行识别操作")
            sg.popup(lang.train_model_warning, text_color="red")
        else:
            print(f"the emotion recognizer is {er}!")
            res_content: list[str] = []
            abs_pathes = get_abs_selected_pathes(speech_folder, selected_files)
            emo_res = []
            # pathes=[]
            import table_show as ts

            for audio in abs_pathes:
                res = er.predict(audio)
                if isinstance(res, list):
                    res = res[0]
                emo_res.append(res)
            print(emo_res, "@{emo_res}")
            print(abs_pathes, "@{abs_pathes}")

            emotion_count_ts = ts.TableShow(
                header=["emotion", "path"], data_lists=[emo_res, abs_pathes]
            )
            print(emotion_count_ts.lists, "@{t.lists}")
            emotion_count_ts.run()

    # 询问是否绘制分析图(以下调用可能会影响FolderBrowse控件的响应)
    if verbose >= 2:
        print("询问绘图环节...")
    dv.data_visualize_events(emotion_count_ts, window=window, event=event)


def refresh_selected_view(window, num_selected_files):
    # 数量
    window[num_selected_files_key].Update(f"({num_selected_files}{lang.files_count_unit})")
    # 内容
    window[selected_files_listbox_key].Update(values=selected_files)


def refresh_viewer(window, speech_folder=None, values=None, delay=1, verbose=1):
    # global speech_folder_path
    # speech_folder_path = Path(values[speech_folder_path_chooser])

    speech_folder_path = Path(speech_folder)
    speech_dir_abs_path = speech_folder_path.absolute()
    speech_dir_abs_posix = speech_dir_abs_path.as_posix()

    # 收集用户的选择
    recursive = values[recursive_checkbox_key]
    short = values[short_path_checkbox_key]
    filter_regex = values[filter_input_key]
    auto_refresh = values[auto_refresh_checkbox_key]

    # print(short, "@{short}🎈")

    if verbose > 1:
        print(speech_dir_abs_posix, "@{dir_abs_posix}")
        print(filter_regex, "@{filter_regex}")

    audio_files = get_audios_regex(
        recursive=recursive,
        short=short,
        filter_regex=filter_regex,
        speech_folder_root=speech_folder_path,
    )
    num_files = len(audio_files)
    # 将扫描到的文件更新到窗口对应组件中,在下一次read方法调用时,画面就会显示新的内容
    window[audio_file_list_key].update(values=audio_files)
    window[num_files_key].update(
        f"{lang.filterd_audios}({num_files} {lang.files_count_unit})"
    )


if __name__ == "__main__":
    pass
    main()

##
