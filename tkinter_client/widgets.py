##
from tkinter import *
from tkinter import ttk

##
root = Tk()

s = ttk.Style()
s.configure("Danger.TFrame", background="red", borderwidth=5, relief="raised")
ttk.Frame(root, width=200, height=200, style="Danger.TFrame").grid()

root.mainloop()
##
import tkinter as tk

root = tk.Tk()
root.geometry("200x200")

# 创建两个标签和两个按钮
label1 = tk.Label(root, text="Label 1")
label2 = tk.Label(root, text="Label 2", anchor="w")
button1 = tk.Button(root, text="Button 1")
button2 = tk.Button(root, text="Button 2", anchor="e")

# 把标签和按钮添加到窗口中
label1.pack()
label2.pack()
button1.pack()
button2.pack()

root.mainloop()

##
import PySimpleGUI as sg

filename = sg.popup_get_file("Enter the file you wish to process")
sg.popup("You entered", filename)
print(filename)

##
import PySimpleGUI as sg  # Part 1 - The import

# Define the window's contents
layout = [
    [sg.Text("What's your name?")],  # Part 2 - The Layout
    [sg.Input()],
    [sg.Button("Ok")],
]

# Create the window
window = sg.Window("Window Title", layout)  # Part 3 - Window Defintion

# Display and interact with the Window
event, values = window.read()  # Part 4 - Event loop or Window.read call

# Do something with the information gathered
print("Hello", values[0], "! Thanks for trying PySimpleGUI")

# Finish up by removing from the screen
window.close()  # Part 5 - Close the Window
##
import PySimpleGUI as sg

# Define the window's contents
layout = [
    [sg.Text("What's your name?")],
    [sg.Input(key="-INPUT-")],
    [sg.Text(size=(40, 1), key="-OUTPUT-")],
    [sg.Button("Ok"), sg.Button("Quit")],
]

# Create the window
window = sg.Window("Window Title", layout)

# Display and interact with the Window using an Event Loop
while True:
    event, values = window.read()
    # See if user wants to quit or window was closed
    if event == sg.WINDOW_CLOSED or event == "Quit":
        break
    # Output a message to the window
    window["-OUTPUT-"].update(
        "Hello " + values["-INPUT-"] + "! Thanks for trying PySimpleGUI",
        text_color="yellow",
    )

# Finish up by removing from the screen
window.close()
##
import PySimpleGUI as sg

layout = [[sg.Button(f"{row}, {col}") for col in range(4)] for row in range(4)]

event, values = sg.Window("List Comprehensions", layout).read(close=True)
# while True:
print(event, values)

# window.close()
##


##
