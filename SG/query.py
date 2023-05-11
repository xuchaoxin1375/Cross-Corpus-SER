import PySimpleGUI as sg

import constants.uiconfig as ufg
import constants.beauty as bt
from SG.multilanguage import get_language_translator

NAME_SIZE = 20

lang = get_language_translator("zh")
sg.theme(bt.ccser_theme)


def name(name):
    dots = NAME_SIZE - len(name) - 2
    return sg.Text(
        name + " " + " " * dots, size=(NAME_SIZE, 1), pad=(0, 0), font="Courier 10"
    )


# if theme:
#     sg.theme(theme)


# Create the layout
def get_query_layout():
    query_layout = [
        [
            sg.Text(
                lang.query_parameter_legend,
                font=("Helvetica", 16),
                pad=((0, 0), (10, 10)),
            )
        ],
        [name(lang.user_name), sg.InputText(key="Username")],
        [name(lang.corpus), sg.InputText(key="Corpus")],
        [name(lang.emotion_feature_prompt), sg.InputText(key="Emotion Feature")],
        [
            name(lang.recognition_alogs_prompt),
            sg.InputText(key="Recognition Algorithm"),
        ],
        [
            name(lang.recognized_file),
            sg.InputText(key="Recognized File"),
        ],
        [sg.Button("Query")],
    ]

    return query_layout


def query_events(event, values, window=None, theme=None):
    # 为了保持风格的一致性,这里可以考虑传入一个window或theme参数
    if theme:
        sg.theme(theme)
    if event == "Query":
        # Process the user input
        username = values["Username"]
        corpus = values["Corpus"]
        feature = values["Emotion Feature"]
        algorithm = values["Recognition Algorithm"]
        file = values["Recognized File"]
        # TODO: Perform the query
        sg.popup("Query Result", "Query successful!")


# Set the window opacity


def run_query():
    query_layout = get_query_layout()
    window = sg.Window("Query Interface", query_layout, resizable=True, finalize=True)
    # window.TKroot.attributes("-alpha", 0.9)
    # Event loop
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == "Cancel":
            break
        query_events(event, values)
    # Close the window
    window.close()


if __name__ == "__main__":
    run_query()
