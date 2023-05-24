import PySimpleGUI as sg
import SG.constants.beauty as bt

R_btn = sg.RButton("Button R", font="Any 25", button_color=("blue", "pink"))
C_btn = sg.Button("Button common", button_color="red on white")
demo_listbox = sg.Listbox(
    values=["a", "b", "c"],
    size=bt.lb_size,
    expand_x=True,
    key="test listbox",
    select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED,
)

layout = [[R_btn], [C_btn], [demo_listbox]]
window = sg.Window(title="sg widget test(display playground)", layout=layout,resizable=True)

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, "exit"):
        break
window.close()
