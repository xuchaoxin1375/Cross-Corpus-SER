import PySimpleGUI as sg
import pandas as pd
from config.MetaPath import recognize_result_dir
from SG.multilanguage import get_language_translator,lang
lang=get_language_translator("zh")
class TableShow():
    def __init__(self,header=None,data_lists=None):
        """将二维列表作为表格数据显示

        Parameters
        ----------
        lists : _type_
            _description_
        """

        self.lists=data_lists
        self.length=len(data_lists[0])
        # 创建表格数据
        self.data_rows = [[l[i] for l in data_lists] for i in range(self.length)]
        # print(self.data_rows,"@{data}")
        # 定义表头
        # header = ["c1","c2"]
        self.header=header#columns

        self.data_df=pd.DataFrame(data=self.data_rows,columns=self.header)
        # 创建表格布局
        save_patient_warning=lang.save_patient_warning
        self.layout = [
            [
                sg.Table(
                    values=self.data_rows,
                    headings=header,
                    max_col_width=100,
                    # background_color="lightblue",
                    auto_size_columns=True,
                    justification="center",
                    num_rows=min(15, len(self.data_rows)),#每次最多显示15条数据
                    expand_x=True,
                    expand_y=True,
                )
            ],
            [sg.Text(lang.close_table_prompt)],
            [sg.Text(lang.save_to_csv_prompt)],
            [sg.Button(lang.save_to_file,tooltip=f"{save_patient_warning}")],

        ]
    def run(self):
        window = sg.Window("result table", self.layout,resizable=True,size=(500,500))
        # 事件循环
        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED:
                break
            # 处理事件
            elif event == "save to file":

                if not recognize_result_dir.exists():
                    recognize_result_dir.mkdir()
                # 将日期和时间格式化为字符串
                from datetime import datetime
                now = datetime.now()
                datetime_str = now.strftime("%Y-%m-%d@%H-%M-%S")
                path=recognize_result_dir/f"recognize_result_{self.length}_{datetime_str}.csv"
                print("Datetime String:", path)

                # data={
                #     self.header[0]:self.lists[0],
                #     self.header[1]:self.lists[1]
                # }
                self.data_df.to_csv(path)
                break

        # 关闭窗口
        window.close()

if __name__=="__main__":
        
    # 创建窗口
    ts=TableShow(header=list("ABC"),data_lists=[[1,2,2],[3,4,4],[5,6,6]])
    ts.run()
    
