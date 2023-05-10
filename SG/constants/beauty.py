import PySimpleGUI as sg

# import constants.uiconfig as ufg
from SG.multilanguage import get_language_translator,lang

import sys

frame_size = (600, 50)
# frame_size=600#给定一个整形数的时候,仅指定宽度,高度被自动设置为1
# frame_size=None
#listbox_size
lb_size = (60, 10)
lb_narrow_size=(20,10)
ml_size = (60, 20)
seperator_color = "blue"
score_ndigits=4
welcom_title_size = (45, 1)
slider_size = (60, 10)
ccser_theme = "Reddit"
title_color = "blue"
res_background_color = "lightblue"
tips_bgc='lightyellow'
result_font = ("Arial", 20, "bold underline")
normal_font=("Arial", 10,"italic")







db_introduction = """
## SpeechDatabases

- 这里主要使用3个语音数据库

### RAVDESS

- [**RAVDESS**](https://zenodo.org/record/1188976) : The **R**yson **A**udio-**V**isual **D**atabase of **E**motional **S**peech and **S**ong that contains 24 actors (12 male, 12 female), vocalizing two lexically-matched statements in a neutral North American accent.

- [RAVDESS Emotional speech audio | Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio?resource=download)

- **File naming convention**

  Each of the 1440 files has a unique filename. The filename consists of a 7-part numerical identifier (e.g., 03-01-06-01-02-01-12.wav). These identifiers define the stimulus characteristics:

  *Filename identifiers*

  - Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
  - Vocal channel (01 = speech, 02 = song).
  - Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
  - Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
  - Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
  - Repetition (01 = 1st repetition, 02 = 2nd repetition).
  - Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

- *Filename example: 03-01-06-01-02-01-12.wav*

  1. Audio-only (03)
  2. Speech (01)
  3. Fearful (06)
  4. Normal intensity (01)
  5. Statement "dogs" (02)
  6. 1st Repetition (01)
  7. 12th Actor (12)
     Female, as the actor ID number is even.

- RAVDESS语料库（Ryerson Audio-Visual Database of Emotional Speech and Song）是一个包含了人类语音和歌曲记录的数据库。该数据库包含了24名演员在读出短语时表现出八种情感状态的语音记录，以及12首歌曲的音频记录。

  RAVDESS语料库的语音记录包含了两种语言（英语和法语），以及四种情感状态的强度（高、中、低和中性）。情感状态包括愤怒、厌恶、恐惧、快乐、悲伤、惊讶和中性。每个演员都会读出两个句子，每个句子表达了四种不同的情感状态。每个短语的长度为三到五个单词。RAVDESS语料库的音频文件格式为WAV，采样率为48kHz，16位量化。

  RAVDESS语料库的歌曲记录包含了12首歌曲，每首歌曲都表达了四种不同的情感状态，包括快乐、悲伤、惊讶和中性。每首歌曲的长度为30秒至1分钟不等，音频文件格式为MP3。

  RAVDESS语料库是一个广泛应用于语音情感识别和分类领域的标准数据集，它已经被广泛应用于语音情感识别和分类算法的开发和评估。该数据库的开放访问使得研究人员可以更方便地进行情感识别和分类算法的开发和评估，同时也为智能语音应用的开发提供了有用的资源。

###  SAVEE

#### Speakers

'DC', 'JE', 'JK' and 'KL' are four male speakers recorded for the SAVEE database


#### Audio data 

Audio files consist of audio WAV files sampled at 44.1 kHz

There are 15 sentences for each of the 7 emotion categories.

The initial letter(s) of the file name represents the emotion class, and the following digits represent the sentence number.

- The letters 'a', 'd', 'f', 'h', 'n', 'sa' and 'su' represent 'anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness' and 'surprise' emotion classes respectively. 

- E.g., 'd03.wav' is the 3rd disgust sentence. 

### EMODB

- [**EMO-DB**](http://emodb.bilderbar.info/docu/) : As a part of the DFG funded research project SE462/3-1 in 1997 and 1999 we recorded a database of emotional utterances spoken by actors. The recordings took place in the anechoic chamber of the Technical University Berlin, department of Technical Acoustics. Director of the project was Prof. Dr. W. Sendlmeier, Technical University of Berlin, Institute of Speech and Communication, department of communication science. Members of the project were mainly Felix Burkhardt, Miriam Kienast, Astrid Paeschke and Benjamin Weiss.

- [EmoDB Dataset | Kaggle](https://www.kaggle.com/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb?resource=download)

- EMODB是爱丁堡多情感数据库（Edinburgh Multi-Emotion Database）的缩写，是一个包含了由演员表演不同情感的音视频记录的数据库。它由爱丁堡大学的研究人员创建，旨在支持情感识别和分类算法的开发和评估。

  该数据库包含了来自英国的10位专业演员（5男5女）的535个音视频记录。每位演员表演了12种不同的情感，包括愤怒、厌恶、恐惧、快乐、悲伤、惊讶等等。这些记录是在一个标准化的环境中进行的，包括标准化的灯光、背景和摄像机角度。

  该数据库已广泛用于情感识别和分类等领域的研究，以及其他相关领域，如语音处理、情感计算和人机交互。该数据库可免费供学术研究使用。

- Code of emotions:

  | letter              | emotion (english) | letter | emotion (german) |
  | ------------------- | ----------------- | ------ | ---------------- |
  | A                   | anger             | W      | Ärger (Wut)      |
  | B                   | boredom           | L      | Langeweile       |
  | D                   | disgust           | E      | Ekel             |
  | F                   | anxiety/fear      | A      | Angst            |
  | H                   | happiness         | F      | Freude           |
  | S                   | sadness           | T      | Trauer           |
  | N = neutral version |                   |        |                  |

- EMODB是一个包含了演员表演不同情感的音视频记录的数据库，其中语音文件的命名方式比较规范，以下是一个示例文件名的分析：

  03a01Wa.wav

  - 03 表示这个音频记录来自第3位演员
  - a01 表示这个音频记录是该演员表演的第1种情感
  - W 表示这个情感是“愤怒”（Angry）的缩写
  - a 表示这个是该情感的第1个副本（第一个表演）
  - .wav 表示这个文件的格式为.wav格式

  因此，这个文件名告诉我们，这个音频记录来自EMODB数据库中的第3位演员，表演的是愤怒情感，并且这是该演员表演愤怒情感的第1个副本。文件的格式为.wav格式。EMODB的语音文件命名方式比较规范，这些信息对于进行情感识别和分类等研究非常有用。

- Additional Information

  Every utterance is named according to the same scheme:

  - Positions 1-2: number of speaker
  - Positions 3-5: code for text
  - Position 6: emotion ( letter stands for german emotion word)
  - Position 7: if there are more than two versions these are numbered a, b, c ....

  Example: 03a01Fa.wav is the audio file from Speaker 03 speaking text a01 with the emotion "Freude" (Happiness).
"""


def h2(
    text="<heading 2>",
    font_family="Arial",
    size=16,
    style="bold",
    style_add="",
    tooltip="",
    **kwargs,
):
    """用于生成二级标题大小的文本段

    Parameters
    ----------
    text : str, optional
        标题内容, by default "<heading 2>"
    font_family : str, optional
        字体家族, by default "Arial"
    size : int, optional
        大小, by default 16
    style : str, optional
        样式,指定的值会覆盖掉默认的样式, by default "bold"
    style_add:str,Optional
        基于默认的样式在追加额外的样式,默认样式得到保留
    Returns
    -------
    sg.Text
        修饰后的文本元素
    """
    style_fields = [style, style_add]
    style = " ".join(list(set(style_fields)))
    if tooltip:
        style_fields.append("italic")
        style = " ".join(style_fields)
        res = sg.Text(
            text=text, font=(font_family, size, style), tooltip=tooltip, **kwargs
        )
    else:
        res = sg.Text(text=text, font=(font_family, size), **kwargs)
    return res


def option_frame(
    title="Option Title",
    layout=[],
    frame_key="option_border",
    size=frame_size,
    title_color=title_color,
    tooltip="",
    expand_x=True,
):
    frame = sg.Frame(
        layout=layout,
        title=title,
        title_color=title_color,
        relief=sg.RELIEF_SUNKEN,
        tooltip=tooltip,
        # size=size if size else frame_size
        key=frame_key,
        expand_x=expand_x,
        # expand_y=True,
    )
    return frame


def result_frame(
    title=lang.result_frame_prompt,
    # result="inputYourContentToHighligt",
    layout=None,
    title_color=title_color,
    frame_key="border",
    expand_x=True,
    visible=True,
    **kwargs,
):
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
    # layout = res_content_layout(layout, expand_x)

    frame = sg.Frame(
        title=title,
        layout=layout,
        title_color=title_color,
        relief=sg.RELIEF_SUNKEN,
        border_width=2,
        expand_x=expand_x,
        # size=bt.size_of_frame
        key=frame_key,
        visible=visible,
        **kwargs
    )

    return frame


def res_content_layout(text, expand_x=True, text_color="red", key=None,justification="c",**kwargs):
    """
    Generates a layout for displaying text content in a GUI window.

    Args:
        text (str): The text to be displayed in the layout.
        expand_x (bool, optional): If True, the text will expand horizontally to fill the available space.
        text_color (str, optional): The color of the text.
        key (str, optional): A key that can be used to reference the text element in the GUI.
        justification (str, optional): The justification of the text within the element.
        **kwargs: Additional arguments that can be passed to the `sg.Text` element.

    Returns:
        list: A layout containing a `sg.Text` element displaying the given text and a horizontal separator.
    """
    layout = [
        [
            sg.Text(
                f"{text}",
                font=result_font,
                background_color=res_background_color,
                text_color=text_color,
                key=key,
                justification=justification,
                expand_x=expand_x,
                **kwargs
            )
        ],
        [sg.HorizontalSeparator()],
    ]

    return layout

def normal_content_layout(text, expand_x=True, text_color="black", key=None,justification="l",**kwargs):
    """
    Generates a layout for displaying text content in a GUI window.

    Args:
        text (str): The text to be displayed in the layout.
        expand_x (bool, optional): If True, the text will expand horizontally to fill the available space.
        text_color (str, optional): The color of the text.
        key (str, optional): A key that can be used to reference the text element in the GUI.
        justification (str, optional): The justification of the text within the element.
        **kwargs: Additional arguments that can be passed to the `sg.Text` element.

    Returns:
        list: A layout containing a `sg.Text` element displaying the given text and a horizontal separator.
    """
    layout = [
        [
            sg.Text(
                f"{text}",
                font=normal_font,
                background_color=tips_bgc,
                text_color=text_color,
                key=key,
                justification=justification,
                expand_x=expand_x,
                **kwargs
            )
        ],
        [sg.HorizontalSeparator()],
    ]

    return layout

if __name__ == "__main__":
    sg.theme()
    # layout_inner = [[sg.Text("demo")]]
    # layout = [[result_frame(title="demo", layout=layout_inner)]]
    # layout=res_content_layout("demo", expand_x=True)
    
    layout=normal_content_layout("demo")
    window = sg.Window("demo of beauty elements", layout,resizable=True)
    window.read()
    window.close()
