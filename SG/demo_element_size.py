import PySimpleGUI as sg

values_list=list("abc")
layout = [
    [sg.ML(size=(10, 2))],
    [sg.ML(size=(50, 10))],
    [sg.LB(values_list,size=(10,2))],
    [sg.LB(values_list,size=(50,10))],
]
window = sg.Window(
    title="check size settings",
    layout=layout,
    # scaling=True
)
window.read()
window.close()
