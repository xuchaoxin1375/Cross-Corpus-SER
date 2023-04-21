import PySimpleGUI as sg
class TableShow():
    def __init__(self,header=None,lists=None):
        """将二维列表作为表格数据显示

        Parameters
        ----------
        lists : _type_
            _description_
        """

        self.lists=lists
        self.length=len(lists[0])
        # 创建表格数据
        data = [[l[i] for l in lists] for i in range(self.length)]
        print(data,"@{data}")
        # 定义表头
        # header = ["c1","c2"]
        self.header=header

        # 创建表格布局
        warning="the save operation will comsume  some time to complete!Be patient!"
        self.layout = [
            [
                sg.Table(
                    values=data,
                    headings=header,
                    max_col_width=100,
                    # background_color="lightblue",
                    auto_size_columns=True,
                    justification="center",
                    num_rows=min(25, len(data)),
                    expand_x=True,
                    expand_y=True,
                )
            ],
            [sg.Text("if you want to recognize the next batch files,please close the window first!\n in the future,the client may be support multiple threads to improve the user experience")],
            [sg.Text("save result to a csv file")],
            [sg.Button(f"save to file",tooltip=f"click to save to a csv file!\n{warning}")],

        ]
    def run(self):
        window = sg.Window("结果表格", self.layout,resizable=True,size=(500,200))
        # 事件循环
        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED:
                break
            # 处理事件
            elif event == "save to file":
                import pandas as pd
                from config.MetaPath import recognize_result_dir
                if not recognize_result_dir.exists():
                    recognize_result_dir.mkdir()
                # 将日期和时间格式化为字符串
                from datetime import datetime
                now = datetime.now()
                datetime_str = now.strftime("%Y-%m-%d@%H-%M-%S")
                path=recognize_result_dir/f"recognize_result_{self.length}_{datetime_str}.csv"
                print("Datetime String:", path)
                
                pd.DataFrame(self.lists,columns=self.header).to_csv(path)
                break

        # 关闭窗口
        window.close()

if __name__=="__main__":
        
    # 创建窗口
    ts=TableShow(header=list("ABC"),lists=[[1,2,2],[3,4,4],[5,6,6]])
    ts.run()
    
