import PySimpleGUI as sg

# import constants.uiconfig as ufg
from SG.multilanguage import get_language_translator,lang

import sys

frame_size = (600, 50)
# frame_size=600#给定一个整形数的时候,仅指定宽度,高度被自动设置为1
# frame_size=None
#listbox_size
lb_size = (60, 10)
input_width=(40,1)
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
columns_limit=4



db_introduction = """
(●'◡'●)
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
    size=(None,None),
    title_color=title_color,
    tooltip="",
    expand_x=True,
):
    frame = sg.Frame(
        layout=layout,
        title=title,
        title_color=title_color,
        # relief=sg.RELIEF_SUNKEN,
        relief=sg.RELIEF_SOLID,
        tooltip=tooltip,
        size=size if size else (None,None),
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
