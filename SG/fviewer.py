##
import os
from pathlib import Path
import PySimpleGUI as sg
import playsound
import re
from config.MetaPath import speech_dbs_dir, savee
import uiconfig as ufg
import table_show as ts
from recognizer.basic import EmotionRecognizer
#将变量设置在事件循环中可能会反复被初始化,这里我们应该放在事件循环的外部
speech_folder_path=Path("./")
selected_files=[]
er:EmotionRecognizer=None
filter_tooltip="""
    the listbox of files allow you to choose one or more files \n using left button of your mouse, 
you can use `Ctrl+Click` to select multiple files(jump to the selected file is allowed too!)

    you can right click after you choose one or more files to do something like these: 
    1.file size
    2.file path(absolute path)
    3.recognize emotion
    4.play file(audio) you choosed
    4.all of above could work in multiple files one by one automatically
"""
selected_files_tooltip="""
you can observe the files your choosed in last listBox
Whether it is a continuous selection or a skip selection, 
these selected files will be tightly arranged and 
the number of files will be displayed at the top
"""

# 创建GUI窗口
audio_viewer_layout = [
    [
        sg.Text("Select a directory:"),
        sg.InputText(default_text="speech db path!",key="speech_folder_path_input"),
        sg.FolderBrowse(initial_folder=speech_dbs_dir /savee,key="speech_folder_path_chooser"),
    ],
    [
        sg.Checkbox(
            "Recursively scan subdirectories", default=True, key="recursive_checkbox"
        )
    ],
    [sg.Text("Filter by regex:"), sg.InputText(key="filter_input",default_text=".*", enable_events=True)],
    [sg.B('filter audio files'),sg.Button(ufg.close)],
    [sg.Text("0 files", key="num_files_text")],
    [
        sg.Listbox(
            values=[],
            size=(50, 10),
            key="audio_files_list",
            enable_events=True,
            bind_return_key=True,
            tooltip=filter_tooltip,
            #定义位于列表中条目的右键菜单内容
            right_click_menu=["", ["Show File Path", "Show File Size",'Play Audio','Emotion Recognize']],
            select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED,
            no_scrollbar=True,
        )
    ],
    [
        sg.Text("Selected audio files:"),
        sg.Text("0 files", key="num_selected_files_text"),
    ],
    [
        sg.Listbox(
            values=[],
            size=(50, 10),
            key="selected_files_list",
           tooltip = selected_files_tooltip,
            # select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED,
        )
    ],
]


# 定义文件大小计算函数
def get_file_size(file_path):
    size = os.path.getsize(file_path)
    size_name = ["Bytes", "KB", "MB", "GB"]
    i = 0
    while size >= 1024 and i < len(size_name) - 1:
        size /= 1024.0
        i += 1
    return f"{size:.2f} {size_name[i]}"

# 事件循环
def get_absolute_path(speech_folder_path, selected_file,verbose=0):
    selected_file= Path(speech_folder_path)/selected_file
    audio_path=selected_file.absolute().as_posix()
    if verbose:
        print(audio_path,"@{audio_path}")
    # print(selected_file,"@{selected_file}")
    return audio_path

def get_abs_selected_pathes(speech_folder_path, selected_files):
    abs_pathes=[]
    for selected_file  in selected_files:
            # values["audio_files_list"]
        abs_path = get_absolute_path(speech_folder_path, selected_file)
        abs_pathes.append(abs_path)
    return abs_pathes
def main():
    sg.theme(ufg.ccser_theme)
    layout = audio_viewer_layout

    window = sg.Window("Audio File Filter", layout,resizable=True)
    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break
        #将变量设置在事件循环中可能会反复被初始化,这里我们应该放在外部
        # speech_folder_path=Path("./")
        # selected_files=[]

        # if event:
        #     print(event)

        # 处理事件
        fviewr_events(window, event, values)

def fviewr_events(window, event, values):
    global selected_files
    global speech_folder_path
        # 处理 "filter_input" 事件
    if event == "filter_input" or event=='filter audio files':
        speech_folder_path = Path(values['speech_folder_path_chooser'])

        dir_posix=Path(speech_folder_path).as_posix()
        print(dir_posix,"@{dir_posix}")

        recursive = values["recursive_checkbox"]
        filter_regex = values["filter_input"]
        if filter_regex == "":
            filter_regex = None
        audio_files = []
        for root, dirs, files in os.walk(speech_folder_path):
            for file in files:
                if file.endswith((".mp3", ".wav", ".ogg")):
                        # 可以设置一个checkbox给用户选择是否显示绝对路径(通常不需要显示绝对目录)
                    file_path = os.path.join(root,file)
                    relative_path_suffix=Path(file_path).relative_to(speech_folder_path)
                        # print(relative_dir,"@{relative_dir}")
                    path=relative_path_suffix.as_posix()
                        # print(path,"@{path}")

                    if not recursive and root != str(speech_folder_path):
                        continue
                    if filter_regex and not re.search(
                            filter_regex, path, re.IGNORECASE
                        ):
                        continue
                    audio_files.append(path)
            if not recursive:
                break
        num_files = len(audio_files)
            # 将扫描到的文件更新到窗口对应组件中,在下一次read方法调用时,画面就会显示新的内容
        window["audio_files_list"].Update(values=audio_files)
        window["num_files_text"].Update(f"Filtered audio files: ({num_files} files)")
        # 处理 "audio_files_list" 事件
    elif event == "audio_files_list":
        selected_files = values["audio_files_list"]
        num_selected_files = len(selected_files)
            #更新选中列表视图控件
            #数量
        window["num_selected_files_text"].Update(
                f"Selected audio files: ({num_selected_files} files)"
            )
            #内容
        window['selected_files_list'].Update(values=selected_files)

        # 处理 "selected_files_list" 事件
        # elif event == "selected_files_list":
        #     selected_files = values["selected_files_list"]
        #     print(selected_files,"@{selected_files}")
        #     num_selected_files = len(selected_files)
        #     window["num_selected_files_text"].Update(f"({num_selected_files} files)")

    elif event == "audio_files_list#Right":
        try:
            file_path = values["audio_files_list"][0]
            menu_choice = sg.popup_menu(["Show File Path", "Show File Size",'Play Audio',"Emotion Recognize"])
            if menu_choice == "Show File Path":
                sg.popup(file_path)
            elif menu_choice == "Show File Size":
                file_size = get_file_size(file_path)
                sg.popup(f"{file_path}:\n{file_size}")
        except IndexError:
            pass
        # 处理 "Show File Path" 事件
    elif event == "Show File Path":
            # for file in values["audio_files_list"]:
        res=[]
        for file in selected_files:
            res.append(get_absolute_path(speech_folder_path, file))
        selected_file_pathes="\n".join(res)
        sg.popup(selected_file_pathes, title="File Path")

        # 处理 "Show File Size" 事件
    elif event == "Show File Size":
            # selected_file = get_abs_selected_pathes(speech_folder_path, selected_files)
        res=[]
        for selected_file in selected_files:
            selected_file=get_absolute_path(speech_folder_path, selected_file)
            file_size = os.path.getsize(selected_file)
            size_str = get_file_size(selected_file)
            sentence=f"The file <{selected_file}>size is {size_str}."
            res.append(sentence)
        res="\n".join(res)
        sg.popup(f"{res}", title="File Size")
        # 处理 "Play Audio" 事件
    elif event == 'Play Audio':
            # selected_file = values['audio_files_list'][0]
        pathes=get_abs_selected_pathes(speech_folder_path, selected_files)
        print(pathes,selected_files)

        from pydub import AudioSegment
        from pydub.playback import play
        for audio_path in pathes:
                # 读取音频文件
            name,ext=os.path.splitext(audio_path)
            print(name,"@{name}",ext,"@{ext}")
                # 播放音频
            audio_file = AudioSegment.from_file(audio_path, format=ext)
            play(audio_file)
    elif event=='Emotion Recognize':
        # print()
        # 为了完成多选文件(成批识别),经过brainstorm,提出以下idea:
        # 委托给ccser_gui模块来处理,通过共享变量来简单通信/创建一个媒介模块来解决相互导入的问题(对于这种简单的场景够用的)
        # 如果出现两个模块相互导入,那么往往要考虑包相互导入的部分中哪些东西抽去到单独的模块中,优化模块的结构
        # 在ccser_gui模块中调用本模块的方法时,采用传参的方式是最直接的通信方式(只不过有些调用参数很多,需要传比较多的参数😂)
        #幸运的是,在python中支持动态添加类(成员属性),可以通过将需要传递的值保留在类的实例中,这样可以减少调用时需要传递的参数(特别时反复用到相关数据时,这更有用)
        #这里的识别应该在训练阶段完成之后才调用的,否则程序应该组织这样跨阶段的行为,提高robustness
        if er ==None:
            print("请先完成识别器训练,然后再执行识别操作")
        else:
            print(f'the emotion recognizer is {er}!')
            res_content:list[str]=[]
            abs_pathes=get_abs_selected_pathes(speech_folder_path, selected_files)
            emo_res=[]
            # pathes=[]
            import table_show as ts
            for audio in abs_pathes:
                emo_res.append(er.predict(audio))
            print(emo_res,"@{emo_res}")
            print(abs_pathes,"@{abs_pathes}")
            t=ts.TableShow(header=["emotion","path"],lists=[emo_res,abs_pathes])
            print(t.lists,"@{t.lists}")
            t.run()


            



if __name__=="__main__":
    main()

##
