##!/usr/bin/env python
import PySimpleGUI as sg
import random
import string

"""
    Basic use of the Table Element
    
    Copyright 2022 PySimpleGUI
"""


# ------ Some functions to help generate data for the table ------
def word(n=5):
    """
    Generate a random string of 10 lowercase letters.

    Returns:
        str: A random string of 10 lowercase letters.
    """
    return "".join(random.choice(string.ascii_lowercase) for i in range(n))


def number(max_val=1000):
    return random.randint(0, max_val)


def make_table_data(num_rows, num_cols):
    """
    生成带有表头的随机表格
    Create a table with the given number of rows and columns.
    The first row will contain random words generated using the word() function
    and the remaining rows will contain a random word followed by
    randomly generated numbers using the number() function.
    Return the created table as a 2D list.

    Parameters:
    -
    num_rows (int): number of rows in the table
    num_cols (int): number of columns in the table

    Returns:
    -
    data (list): a 2D list representing the created table.
    """
    data = [[j for j in range(num_cols)] for i in range(num_rows)]

    # 生成n个单词(生成随机表头)
    data[0] = [word() for _ in range(num_cols)]
    # 从data[1]开始的所有行的都是由一个单词打头后续的列为随机生成的数字
    for i in range(1, num_rows):
        # 利用星号表达式解包
        data[i] = [word(), *[number() for i in range(num_cols - 1)]]
    # 利用pandas查看生成的数据结果
    # import pandas as pd
    # df = pd.DataFrame(data)
    # print(df)
    return data


# ------ Make the Table Data ------
data = make_table_data(num_rows=15, num_cols=6)
# 为每个表头字段加上2个点
headings = [str(data[0][x]) for x in range(len(data[0]))]

# ------ Window Layout ------
layout = [
    [
        sg.Table(
            values=data[1:],  # 从表格的第1行开始到最后一行的部分
            headings=headings,
            max_col_width=25,  # max_col_width和auto_size_column会相互影响吗?
            auto_size_columns=True,
            # cols_justification=('left','center','right','c', 'l', 'bad'),       # Added on GitHub only as of June 2022
            display_row_numbers=True,
            justification="center",
            num_rows=10,
            alternating_row_color="lightblue",
            key="-TABLE-",
            selected_row_colors="red on yellow",
            enable_events=True,
            expand_x=True,
            expand_y=True,
            vertical_scroll_only=False,
            enable_click_events=True,  # Comment out to not enable header and other clicks
            tooltip="This is a table",
        )
    ],
    [sg.Button("Read"), sg.Button("Double"), sg.Button("Change Colors")],
    [sg.Text("Read = read which rows are selected")],
    [sg.Text("Double = double the amount of data in the table")],
    [
        sg.Text("Change Colors = Changes the colors of rows 8 and 9"),
        #  sg.Sizegrip()
    ],
]


def main():
    # ------ Create Window ------

    window = sg.Window(
        "The Table Element",
        layout,
        # ttk_theme='clam',
        # font='Helvetica 25',
        resizable=True,
    )

    # ------ Event Loop ------
    while True:
        event, values = window.read()
        print(event, values)
        if event == sg.WIN_CLOSED:
            break
        if event == "Double":
            for i in range(1, len(data)):
                data.append(data[i])
            window["-TABLE-"].update(values=data[1:])
        elif event == "Change Colors":
            window["-TABLE-"].update(row_colors=((8, "white", "red"), (9, "green")))

    window.close()


if __name__ == "__main__":
    main()
    # table=make_table_data(3,5)
    # print(table)
    # print(table[1:])
    # print(data[1:][:])
