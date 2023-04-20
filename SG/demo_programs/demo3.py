import PySimpleGUI as sg
# sg.theme('Dark Brown')
def make_window1(theme=None):
    if theme:
        sg.theme(theme)
    layout = [[sg.Text('Theme Browser')],
            [sg.Text('Click a Theme color to see demo window')],
            [sg.Listbox(values=sg.theme_list(), size=(20, 12), key='-LIST-', enable_events=True,select_mode='extended')],
            [sg.Button('set theme')]]
    window = sg.Window('Theme Browser', layout)
    return window
def main1():
    window=make_window1()
    while True:  # Event Loop
        event, values = window.read()
        if event in (None, 'Exit'):
            break
        elif event=='-LIST-':
            print(event,"[I],theme item clicked",values['-LIST-'])

        elif(event=='set theme'):
            theme=values['-LIST-'][0]
            window.close()
            window=make_window1(theme)
            #使用refresh无法更新主题,需要close原来的window后重建
            # sg.theme(theme)
            # window.refresh()
    window.close()

def main2():

    options = ['Option 1', 'Option 2', 'Option 3']

    layout = [[sg.Text('Select options:')],
            [sg.Listbox(values=options, size=(30, 5), select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED)],
            [sg.Button('Ok'), sg.Button('Cancel')]]

    window = sg.Window('Window Title', layout)

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Cancel'):
            break
        elif event == 'Ok':
            selected_options = values[0]
            sg.popup('Selected options:', selected_options)

    window.close()




    
if __name__=="__main__":
    # main1()
    main2()