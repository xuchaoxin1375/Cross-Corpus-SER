def demo1():
    layout = [
        [sg.Text("My one-shot window.")],
        [sg.InputText()],
        [sg.Submit(), sg.Cancel()],
    ]
    # window.read()打开窗口
    window = sg.Window("Window Title", layout)

    event, values = window.read()
    window.close()
    text_input = values[0]
    print(event, values)  # Submit {0: ''}
    # 使用popup直接弹出一个提示窗口
    sg.popup("You entered", text_input)


def demo2():
    layout = [
        [sg.Text("My one-shot window.")],
        [sg.InputText(key="-IN-")],
        [sg.Submit(), sg.Cancel()],
    ]

    window = sg.Window("Window Title", layout)

    event, values = window.read()

    window.close()

    text_input = values["-IN-"]
    print(event, values)  # Submit {'-IN-': ''}
    sg.popup("You entered", text_input)


import PySimpleGUI as sg


def demo3():
    sg.theme("DarkAmber")  # Keep things interesting for your users

    layout = [
        [sg.Text("Persistent window")],
        [sg.Input(key="-IN-")],
        [sg.Button("Read"), sg.Exit()],
    ]

    window = sg.Window("Window that stays open", layout)

    while True:  # The Event Loop
        event, values = window.read()
        print(event, values)
        if event == sg.WIN_CLOSED or event == "Exit":
            break

    window.close()


def demo4():
    sg.theme("BluePurple")

    layout = [
        [
            sg.Text("Your typed chars appear here:"),
            sg.Text(size=(15, 1), key="-OUTPUT-"),
        ],
        [sg.Input(key="-IN-")],
        [sg.Button("Show"), sg.Button("Exit")],
    ]

    window = sg.Window("Pattern 2B", layout)
    # print(window,type(window))

    while True:  # Event Loop
        event, values = window.read()
        print(event, values)
        if event == sg.WIN_CLOSED or event == "Exit":
            break
        elif event == "Show":
            # Update the "output" text element to be the value of "input" element
            window["-OUTPUT-"].update(values["-IN-"])

    window.close()


def demo5():
    layout = [
        [sg.Text("Click the button to do something")],
        [sg.Button("Click me", tooltip="Press this button to do something")],
    ]

    window = sg.Window("My Window", layout)

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break

    window.close()


def demo6():
    """
    Allows you to "browse" through the Theme settings.  Click on one and you'll see a
    Popup window using the color scheme you chose.  It's a simple little program that also demonstrates
    how snappy a GUI can feel if you enable an element's events rather than waiting on a button click.
    In this program, as soon as a listbox entry is clicked, the read returns."""

    sg.theme("Dark Brown")

    layout = [
        [sg.Text("Theme Browser")],
        [sg.Text("Click a Theme color to see demo window")],
        [
            sg.Listbox(
                values=sg.theme_list(), size=(20, 12), key="-LIST-", enable_events=True
            )
        ],
        [sg.Button("Exit")],
    ]

    window = sg.Window("Theme Browser", layout)

    while True:  # Event Loop
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "Exit"):
            break
        sg.theme(values["-LIST-"][0])
        sg.popup_get_text("This is {}".format(values["-LIST-"][0]))

    window.close()


def demo_checkbox():
    sg.theme("DarkBlue")

    layout = [
        [sg.Text("请选择您喜欢的水果：")],
        [
            sg.Checkbox("苹果", key="apple"),
            sg.Checkbox("香蕉", key="banana"),
            sg.Checkbox("橙子", key="orange"),
        ],
        [sg.Button("确定"), sg.Button("取消")],
    ]

    window = sg.Window("Checkbox示例", layout)

    while True:
        event, values = window.read()
        print(event, values)
        if event in (None, "取消"):
            break
        elif event == "确定":
            selected_fruits = [fruit for fruit, selected in values.items() if selected]
            if selected_fruits:
                sg.popup("您选择了：{}".format(", ".join(selected_fruits)))
            else:
                sg.popup("您没有选择任何水果！")

    window.close()


def demo_choose_file():
    sg.theme("DarkBlue")

    # 定义GUI布局
    file_choose_layout = [
        [sg.Text("请选择一个文件：")],
        [sg.Input(), sg.FileBrowse()],
        [sg.Button("确定"), sg.Button("取消")],
    ]

    # 创建窗口
    window = sg.Window("文件选择示例", file_choose_layout)

    # 事件循环
    while True:
        event, values = window.read()
        if event in (None, "取消"):
            break
        elif event == "确定":
            selected_file = values[0]
            sg.popup("您选择的文件是：{}".format(selected_file))

    # 关闭窗口
    window.close()


def demo_multi_choose():
    sg.theme("DarkBlue")

    # 定义GUI布局
    features_choose_layout = [
        [sg.Text("请选择一个或多个特征：")],
        [
            sg.Checkbox("MFCC", key="mfcc"),
            sg.Checkbox("Mel", key="mel"),
            sg.Checkbox("Contrast", key="contrast"),
        ],
        [sg.Checkbox("Chromagram", key="chroma"), sg.Checkbox("Tonnez", key="tonnez")],
        [sg.Button("确定"), sg.Button("取消")],
    ]

    # 创建窗口
    window = sg.Window("特征选择示例", features_choose_layout)

    # 事件循环
    while True:
        event, values = window.read()
        if event in (None, "取消"):
            break
        elif event == "确定":
            selected_features = [key for key, value in values.items() if value]
            sg.popup("您选择的特征是：{}".format(", ".join(selected_features)))

    # 关闭窗口
    window.close()


def demoCombo():
    databases = ["emodb", "ravdess", "savee"]

    # Define the layout of the dialog
    db_choose_layout = [
        [sg.Text("Select the training database")],
        [sg.Combo(databases, key="train_db")],
        [sg.Text("Select the testing database")],
        [sg.Combo(databases, key="test_db")],
        [sg.Button("OK"), sg.Button("Cancel")],
    ]

    # Create the dialog
    window = sg.Window("Select Databases", db_choose_layout)

    # Initialize the selected databases list
    selected_databases = []

    # Loop until the user clicks the OK button or closes the dialog
    while True:
        # 显示窗口
        # 当用户点击相关按钮等事件被提交,才会从执行下一条语句
        # 通常可以通过点击OK按钮来提交(以进行下一步)
        event, values = window.read()
        # 判断提交的事件是否为None或Cancel,如果不是,说明用户的提交内容是完整且合法的,否则要打断循环事件
        if event in (None, "Cancel"):
            break
        # 如果执行到这里,可以读取用户提交的event和value

        # Check if the user has already selected the same database

        # Add the selected databases to the list
        selected_databases.append(values["train_db"])
        selected_databases.append(values["test_db"])

        # Remove the selected databases from the options
        databases.remove(values["train_db"])
        databases.remove(values["test_db"])

        # If the user has selected two databases, exit the loop
        if len(selected_databases) == 2:
            break

    # Close the dialog
    window.close()

    # Print the selected databases
    print("Training database:", selected_databases[0])
    print("Testing database:", selected_databases[1])


def preprocess_1(file_path):
    # 第一个预处理操作
    print("执行预处理操作1：{}".format(file_path))


def preprocess_2(file_path):
    # 第二个预处理操作
    print("执行预处理操作2：{}".format(file_path))


def preprocess_3(file_path):
    # 第三个预处理操作
    print("执行预处理操作3：{}".format(file_path))


def demo_checkboxes(preprocess_1, preprocess_2, preprocess_3):
    sg.theme("DarkBlue")

    # 定义GUI布局
    draw_layout = [
        [sg.Text("请选择一个文件：")],
        [sg.Input(), sg.FileBrowse()],
        [
            sg.Checkbox("预处理操作1", key="preprocess_1"),
            sg.Checkbox("预处理操作2", key="preprocess_2"),
            sg.Checkbox("预处理操作3", key="preprocess_3"),
        ],
        [sg.Button("确定"), sg.Button("取消")],
    ]

    # 创建窗口
    window = sg.Window("预处理示例", draw_layout)

    # 事件循环
    while True:
        event, values = window.read()
        if event in (None, "取消"):
            break
        elif event == "确定":
            selected_file = values[0]
            if values["preprocess_1"]:
                preprocess_1(selected_file)
            if values["preprocess_2"]:
                preprocess_2(selected_file)
            if values["preprocess_3"]:
                preprocess_3(selected_file)

    # 关闭窗口
    window.close()


def demo_event_value():
    # 定义GUI布局
    layout = [
        [sg.Text("请输入您的姓名：")],
        [sg.InputText()],
        [sg.Checkbox("check item:show the use of key", key="-checkbox-")],
        [sg.Button("确定")],
    ]

    # 创建GUI窗口
    window = sg.Window("示例:event和value变量的使用", layout)

    # 处理GUI事件和用户输入
    while True:
        event, values = window.read()
        print("@{event}:")
        print(event)
        print("@{values}:")
        print(values)

        if event == sg.WIN_CLOSED:
            break

        if event == "确定":
            name = values[0]
            index_value = values[0]
            key_checked_value = values["-checkbox-"]
            sg.popup(
                f"您好，{name}！欢迎使用PySimpleGUI。\n\
                     {event=},{index_value=},{key_checked_value=}"
            )

    # 关闭GUI窗口
    window.close()




if __name__ == "__main__":
    # demo6()
    # demo_checkbox()
    # demo_choose_file()
    # demo_multi_choose()
    # demo_checkboxes(preprocess_1, preprocess_2, preprocess_3)
    # twice_choose()
    # demo_event_value()
    # demoCombo()
    multiple_inputs()
