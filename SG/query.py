import PySimpleGUI as sg
NAME_SIZE = 20
import uiconfig as ufg
sg.theme(ufg.ccser_theme)
def name(name):
    dots = NAME_SIZE-len(name)-2
    return sg.Text(name + ' ' + ' '*dots, size=(NAME_SIZE,1), pad=(0,0), font='Courier 10')
# if theme:
#     sg.theme(theme)

# Create the layout
query_layout = [
    [sg.Text("Query Parameters", font=("Helvetica", 16), pad=((0, 0), (10, 10)))],
    [name("Username:"), sg.InputText(key="Username")],
    [name("Corpus:"), sg.InputText(key="Corpus")],
    [name("Emotion Feature:"), sg.InputText(key="Emotion Feature")],
    [name("Recognition Algorithm:"), sg.InputText(key="Recognition Algorithm")],
    [name("Recognized File:"), sg.InputText(key="Recognized File")],
    [sg.Button("Query")],
]

def query_events(event, values,window=None,theme=None):
    #为了保持风格的一致性,这里可以考虑传入一个window或theme参数
    if theme:
        sg.theme(theme)
    if event == "Query":
        # Process the user input
        username = values['Username']
        corpus = values['Corpus']
        feature = values['Emotion Feature']
        algorithm = values['Recognition Algorithm']
        file = values['Recognized File']
        # TODO: Perform the query
        sg.popup("Query Result", "Query successful!")

# Set the window opacity

def run_query():
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


if __name__=="__main__":
    run_query()


