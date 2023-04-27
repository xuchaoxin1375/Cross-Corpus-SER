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

# from recognizer.basic import EmotionRecognizer
import data_visualization as dv

# 主题设置说明:当主题设置语句防止在程序的末尾时可能是无效的
# 猜测sg.theme()设置完主题后,后续在调用sg的元素创建方法才会有相应主题的配色
# 如果控件都已经创建好了才开始调用sg.theme()修改配色,那来不及起作用了
sg.theme(bt.ccser_theme)
# 常量
listbox_default_value_tip = "hover your mouse in this listbox area to check tooltips!"
audio_listbox_values = [
    "click filter or input regex to scan audio file!",
    listbox_default_value_tip,
]

# 将变量设置在事件循环中可能会反复被初始化,这里我们应该放在事件循环的外部
# speech_folder_path=Path("./")
speech_folder = speech_dbs_dir
emodb_dir = speech_dbs_dir / "emodb/wav"
audio_exts = (".wav", ".mp3", ".ogg")
savee_dir = speech_dbs_dir / "savee/AudioData"

selected_files = []
# er: EmotionRecognizer = None
er = None
t: ts.TableShow = None
filter_tooltip = """
    the listbox of files allow you to choose one or more files \n using left button of your mouse, 
you can use `Ctrl+Click` to select multiple files(jump to the selected file is allowed too!)

    you can right click after you choose one or more files to do something like these: 
    1.file size
    2.file path(absolute path)
    3.recognize emotion
    4.play file(audio) you choosed
    *.all of above could work in multiple files one by one automatically
"""
selected_files_tooltip = """
you can observe the files your choosed in last listBox
Whether it is a continuous selection or a skip selection, 
these selected files will be tightly arranged and 
the number of files will be displayed at the top
"""
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
        if filter_regex:
            s = re.search(filter_regex, str(path), re.IGNORECASE)
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
speech_folder_path_input_key = "speech_folder_path_input"
speech_folder_path_chooser_key = "speech_folder_path_chooser"
default_folder_file_list = get_audios_regex(
    recursive=True, speech_folder_root=speech_dbs_dir, short=True
)

# print(default_folder_file_list[:10])
##
len_default_folder_file_list = len(default_folder_file_list)

right_click_menu = [
    "",
    ["Show File Path", "Show File Size", "Play Audio", "Emotion Recognize"],
]

audio_viewer_layout = [
    [sg.Text("Select a directory:")],
    [
        sg.InputText(
            default_text=speech_folder,
            key=speech_folder_path_input_key,
            tooltip="you can paste or type a dir path!\n or use the right side Browse button to choose a dir",
        ),
        sg.FolderBrowse(
            initial_folder=folder_browse_init_dir,
            button_text="folder browse",
            change_submits=True,
            key=speech_folder_path_chooser_key,
            target=speech_folder_path_input_key,
            # enable_events=True,
            tooltip=f"choose a folder you want to do SER,\nthe default folder is {speech_folder}",
        ),
    ],
    [sg.B(confirm_folder_selected_key)],
    [
        sg.Input(
            default_text="select multiple files,which will be shown here ",
            key=files_browsed_key,
        ),
        # sg.Text(text="files selected by filesBrowse will be shown \n in the listbox below"),
        sg.FilesBrowse(
            target=files_browsed_key,
            key="FilesBrowse",
            enable_events=True,
            change_submits=True,
        ),
        sg.OK(key="confirm files selected"),
    ],
    [
        sg.Text("current directory:"),
        sg.Text(f"{speech_folder}", key="current_dir"),
    ],
    [
        sg.Checkbox(
            text="Recursively scan subdirectories",
            default=True,
            key=recursive_checkbox_key,
            enable_events=True,
        ),
        sg.Checkbox(
            text="auto refresh",
            default=False,
            key=auto_refresh_checkbox_key,
            enable_events=True,
        ),
        sg.Checkbox(
            text="short path", default=True, key=short_path_checkbox_key, enable_events=True
        ),
    ],
    [
        sg.Text("Filter by regex:"),
        sg.InputText(key="filter_input", default_text="", enable_events=True),
    ],
    [
        sg.B(filter_audios_key, tooltip="click to manual refresh the files listbox"),
        sg.Button(ufg.close),
    ],
    [sg.Text(f"{len_default_folder_file_list} files", key="num_files_text")],
    [
        sg.Listbox(
            values=default_folder_file_list,
            # size=(50, 10),
            size=bt.lb_size,
            key=audio_file_list_key,
            enable_events=True,
            bind_return_key=True,
            tooltip=filter_tooltip,
            # 定义位于列表中条目的右键菜单内容
            right_click_menu=right_click_menu,
            select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED,
            no_scrollbar=True,
        )
    ],
    [
        sg.Text("Selected audio files:"),
        sg.Text(f"0 files", key=num_selected_files_key),
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
    layout = audio_viewer_layout

    window = sg.Window("Audio File Filter", layout, resizable=True)
    while True:
        event, values = window.read()
        print(event, "@{event} main")
        if event in (sg.WINDOW_CLOSED, ufg.close):
            break
        else:
            # 处理事件(小心,如果下面的函数编写不当,可能使得某些控件不能够正常工作)
            # 例如,FolderBrowser生成的按钮点击无法呼出系统的资源管理器(或者需要反复点击)
            pass
            fviewer_events(window, event, values)

    window.close()


def fviewer_events(window, event=None, values=None, verbose=1):
    global selected_files
    global speech_folder
    global t
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
            print(path, "was confirmed")
            if Path(path).exists():
                speech_folder = path
                # 更新当前speech_path控件
                window["current_dir"].Update(path)
                # 更新文件列表视图
                refresh_viewer(window, speech_folder=path, values=values)
            else:
                sg.popup_error(f"{path} not exist!")
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
    elif event == "Show File Path":
        res = []
        for file in selected_files:
            res.append(get_absolute_path(speech_folder, file))
        selected_file_pathes = "\n".join(res)
        sg.popup(selected_file_pathes, title="File Path")

        # 处理 "Show File Size" 事件
    elif event == "Show File Size":
        # selected_file = get_abs_selected_pathes(speech_folder_path, selected_files)
        res = []
        for selected_file in selected_files:
            selected_file = get_absolute_path(speech_folder, selected_file)
            file_size = os.path.getsize(selected_file)
            size_str = get_file_size(selected_file)
            sentence = f"The file <{selected_file}>size is {size_str}."
            res.append(sentence)
        res = "\n".join(res)
        sg.popup(f"{res}", title="File Size")
        # 处理 "Play Audio" 事件
    elif event == "Play Audio":
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
    elif event == "Emotion Recognize":
        # print()
        # 为了完成多选文件(成批识别),经过brainstorm,提出以下idea:
        # 委托给ccser_gui模块来处理,通过共享变量来简单通信/创建一个媒介模块来解决相互导入的问题(对于这种简单的场景够用的)
        # 如果出现两个模块相互导入,那么往往要考虑包相互导入的部分中哪些东西抽去到单独的模块中,优化模块的结构
        # 在ccser_gui模块中调用本模块的方法时,采用传参的方式是最直接的通信方式(只不过有些调用参数很多,需要传比较多的参数😂)
        # 幸运的是,在python中支持动态添加类(成员属性),可以通过将需要传递的值保留在类的实例中,这样可以减少调用时需要传递的参数(特别时反复用到相关数据时,这更有用)
        # 这里的识别应该在训练阶段完成之后才调用的,否则程序应该组织这样跨阶段的行为,提高robustness
        if er == None:
            print("请先完成识别器训练,然后再执行识别操作")
            sg.popup("please train the SER model and then try angin!")
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

            t = ts.TableShow(header=["emotion", "path"], data_lists=[emo_res, abs_pathes])
            print(t.lists, "@{t.lists}")
            t.run()

    # 询问是否绘制分析图(以下调用可能会影响FolderBrowse控件的响应)
    if verbose >= 2:
        print("询问绘图环节...")
    dv.data_visualize_events(t, window=window, event=event)


def refresh_selected_view(window, num_selected_files):
    # 数量
    window[num_selected_files_key].Update(
        f"Selected audio files: ({num_selected_files} files)"
    )
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
    auto_refresh=values[auto_refresh_checkbox_key]

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
    window[num_files_key].update(f"Filtered audio files: ({num_files} files)")


if __name__ == "__main__":
    pass
    main()

##
