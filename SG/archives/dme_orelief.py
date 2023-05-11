import PySimpleGUI as sg

sg.theme('LightGrey1')

def make_text(text, relief):
    return sg.Text(text, relief=relief, size=(10, 2))

layout = [
    [make_text('Raised', sg.RELIEF_RAISED), make_text('Sunken', sg.RELIEF_SUNKEN), make_text('Flat', sg.RELIEF_FLAT)],
    [make_text('Ridge', sg.RELIEF_RIDGE), make_text('Groove', sg.RELIEF_GROOVE), make_text('Solid', sg.RELIEF_SOLID)]
]

window = sg.Window('Relief Demo', layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break

window.close()