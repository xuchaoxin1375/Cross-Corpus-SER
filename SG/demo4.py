import PySimpleGUI as sg

# # res=sg.popup("hello")
# long_text="""
# # layout = [[sg.popup("hello")]]
# # window.read()
# # window.close()
# """
# res=sg.popup_scrolled(long_text)
# print(res)
##

# layout=[
#     [sg.Input("enable_events=True",key="in1",enable_events=True)],
#     [sg.Input("enable_evenets=False",key="in2",enable_events=False)],
#     [sg.B("OK")]
# ]
# window=sg.Window("My App",layout)
# while True:
#     e,v=window.read()
#     print(e,v)
#     if e==sg.WIN_CLOSED:
#         break

# ##
# long_text="""

# ░█████╗░░█████╗░░██████╗███████╗██████╗░░░░░░░░█████╗░██╗░░░░░██╗███████╗███╗░░██╗████████╗
# ██╔══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗░░░░░░██╔══██╗██║░░░░░██║██╔════╝████╗░██║╚══██╔══╝
# ██║░░╚═╝██║░░╚═╝╚█████╗░█████╗░░██████╔╝█████╗██║░░╚═╝██║░░░░░██║█████╗░░██╔██╗██║░░░██║░░░
# ██║░░██╗██║░░██╗░╚═══██╗██╔══╝░░██╔══██╗╚════╝██║░░██╗██║░░░░░██║██╔══╝░░██║╚████║░░░██║░░░
# ╚█████╔╝╚█████╔╝██████╔╝███████╗██║░░██║░░░░░░╚█████╔╝███████╗██║███████╗██║░╚███║░░░██║░░░
# ░╚════╝░░╚════╝░╚═════╝░╚══════╝╚═╝░░╚═╝░░░░░░░╚════╝░╚══════╝╚═╝╚══════╝╚═╝░░╚══╝░░░╚═╝░░░

# """
# sg.popup_scrolled(long_text)
# window.close()
##
import PySimpleGUI as sg      

sg.theme('LightGreen')      
sg.set_options(element_padding=(0, 0))      

# ------ Menu Definition ------ #      
menu_def = [['File', ['Open', 'Save', 'Exit'  ]],      
            ['Edit', ['Paste', ['Special', 'Normal', ], 'Undo'], ],      
            ['Help', 'About...'], ]      

# ------ GUI Defintion ------ #      
layout = [      
    [sg.Menu(menu_def, )],      
    [sg.Output(size=(60, 20))]      
            ]      

window = sg.Window("Windows-like program", layout, default_element_size=(12, 1), auto_size_text=False, auto_size_buttons=False,      
                    default_button_element_size=(12, 1))      

# ------ Loop & Process button menu choices ------ #      
while True:      
    event, values = window.read()      
    if event == sg.WIN_CLOSED or event == 'Exit':      
        break      
    print('Button = ', event)      
    # ------ Process menu choices ------ #      
    if event == 'About...':      
        sg.popup('About this program', 'Version 1.0', 'PySimpleGUI rocks...')      
    elif event == 'Open':      
        filename = sg.popup_get_file('file to open', no_window=True)      
        print(filename)  