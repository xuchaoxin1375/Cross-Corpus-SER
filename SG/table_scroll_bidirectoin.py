""" 
这个demo假设要根据一系列函数名作为侧边大纲(目录),在内容相对窗口过长的时候支持垂直滚动
而主体是一个支持双向滚动的表格(当内容相对于窗口过大过多的时候)
"""
import PySimpleGUI as sg

header_list = [
    "Function",
    "Starting Address",
    "Basic Blocks",
    "Instructions",
    "Cyclomatic Complexity",
    "Jilb's Metric",
    "ABC Metric",
    "Halstead Estimated Length",
    "Halstead Volume",
    "Halstead Difficulty",
    "Halstead Effort",
    "Halstead Time",
    "Halstead Bugs",
    "Harris Metric",
    "Oviedo Metric",
    "Chepin Metric",
    "Card & Glass Metric",
    "Henry & Kafura Metric",
    "Cocol Metric",
    "Hybrid General Complexity",
]
import random

def create_data_fake(header_list,table_data):
    """ 
    Create 20 rows of randomly generated data for a given header list.
      Each row will have the same length as the header list. 
      The data will be rounded to 3 decimal places. The table_data parameter is a list that will be modified to include the generated data.
    """
    n_cols=len(header_list)
    print(id(table_data),"@in the create_data_fake::before assign new value,this comes from argument")
    table_data=[
        [
           round(random.random(),3) for r_ in range(20)
        ]
        for c_ in range(n_cols)
    ]
    print(id(table_data),"@in the create_data_fake::after assign new value,this will be return")
    return table_data
def create_data_quick(header_list):
    """ 
    Create 20 rows of randomly generated data for a given header list.
      Each row will have the same length as the header list. 
      The data will be rounded to 3 decimal places. The table_data parameter is a list that will be modified to include the generated data.
    """
    n_cols=len(header_list)
    table_data=[
        [
           round(random.random(),3) for r_ in range(20)
        ]
        for c_ in range(n_cols)
    ]
    return table_data
function_names = [
    "main",
    "printf",
    "sin",
    "day_of_year",
    "month_day",
    "swap",
    "qsort",
    "fopen",
    "fclose",
    "fscanf",
    "fgetc",
    "getchar",
    "putc",
    "fwrite",
    "fseek",
    "strcpy",
    "strcat",
    "strcmp",
    "strchr",
    "atof",
]


def make_layouts(header_list, table_data, function_names):
    """
    Generate a layout for a PySimpleGUI window based on header_list, table_data,
    and function_names.

    Args:
    - header_list (list of str): List of column names for the table.
    - table_data (list of list of str): List of rows of data for the table.
    - function_names (list of str): List of function names to display in the first
        column.

    Returns:
    - layout (list of list of PySimpleGUI elements): Layout for the PySimpleGUI window.
    """
    print(header_list[:1],"@{header_list[:1]}")
    toc_col_layout = [
        [
            sg.Table(
                values=function_names,
                headings=header_list[:1],#headings参数接收列表,即使只有一个字符串,也要使用list封装起来
                max_col_width=25,
                auto_size_columns=True,
                display_row_numbers=False,
                justification="c",
                # hide_vertical_scroll=True,
                alternating_row_color="#626366",
                num_rows=min(len(function_names), 20),
            )
        ]
    ]
    window_col_layout = [
        [
            sg.Table(
                values=table_data,
                headings=header_list[1:],
                max_col_width=25,

                auto_size_columns=True,

                justification="center",
                vertical_scroll_only=False,
                expand_x=True,
                # alternating_row_color="#626366",
                num_rows=5,
            )
        ]
    ]
    import copy 
    toc_col_layout1=copy.deepcopy(toc_col_layout)
    window_col_layout1=copy.deepcopy(window_col_layout)
    layout = [
        # [
        #     sg.Col(toc_col_layout, vertical_alignment="top"),
        #     sg.Col(window_col_layout, vertical_alignment="top"),
        # ],
        # [sg.Button("Close")],
        # *toc_col_layout1,
        *window_col_layout1,
    ]

    return layout




def main():
    """ This function generates a GUI window to display a table of data. It takes in a headerlist and tabledata as arguments to create the data, and a function_names list to make the layouts. The function uses PySimpleGUI to create a window with a "Close" button to terminate the program. The function returns when the window is closed via the "x" button or selecting the "Close" button. :return: None """

    table_data=[]
    table_data=create_data_quick(header_list)
    print(table_data,"@{table_data}")
    
    layout = make_layouts(header_list, table_data, function_names)
    # Generated the gui window to display the table
    window = sg.Window(
        "Function Metrics",
        layout,
        resizable=True,
        size=(600, 300),
        font="AndaleMono 16",
    )

    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        # 	del values
        # End the function if the windows is closed via the "x" button or selecting the "Close" button
        if event == sg.WIN_CLOSED or event == "Close":
            break

    # Terminate the window, which will cause the function to return and the program to end
    window.close()
if __name__=="__main__":
    main()
