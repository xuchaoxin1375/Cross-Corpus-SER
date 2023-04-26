import PySimpleGUI as sg

layout_pane = [
    [sg.Button("Button 1")],
    [sg.Button("Button 2")],
    [sg.InputText()]
]

layout = [
    [sg.Text("This is a sample window with a Pane element")],
    [sg.Pane(layout_pane, size=(300, 200))],
    [sg.Button("OK"), sg.Button("Cancel")]
]

window = sg.Window("Sample Window with Pane", layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == "Cancel":
        break

window.close()