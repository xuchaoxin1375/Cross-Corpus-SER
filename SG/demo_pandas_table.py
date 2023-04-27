import PySimpleGUI as sg
import pandas as pd
import SG.constants.beauty as bt

sg.theme(bt.ccser_theme)
class TablePandas():
    def __init__(self,df=None) -> None:
        if df is None:
            # 创建一个 Pandas 数据帧
            demo_data = {
                "Name": ["Alice", "Bob", "Charlie", "David"],
                "Age": [25, 30, 35, 40],
                "Salary": [50000, 60000, 70000, 80000],
            }

            df = pd.DataFrame(demo_data)
        self.df=df
    
    # 创建 PySimpleGUI 窗口布局
    def create_table_window(self,df):
        layout = [
            [
                sg.Table(
                    values=df.values.tolist(),
                    headings=df.columns.tolist(),
                    max_col_width=25,
                    auto_size_columns=True,
                    justification="center",
                    num_rows=min(25, len(df)),
                )
            ],
            # [sg.Button("Exit")],
        ]

        return layout
    def get_confution_matrix_window(self,df=None):
        layout = self.create_table_window(df)
        window = sg.Window("Pandas Table Viewer", layout)
        return window
    # def show_confution_matrix_window(df):
    #     window=get_confution_matrix_window(df)
        

    def show_confution_matrix_window(self,df=None):
        # 创建 PySimpleGUI 窗口
        # window = sg.Window("Pandas Table Viewer", layout)
        df=df if df else self.df
        window=self.get_confution_matrix_window(df=df)
        # 处理事件循环
        while True:
            event, values = window.read()
            if event in (sg.WINDOW_CLOSED,):
                break

        # 关闭 PySimpleGUI 窗口
        window.close()
if __name__=="__main__":
    tp=TablePandas()
    tp.show_confution_matrix_window()
    # show_confution_matrix_window(df)