##

import re
from config.MetaPath import trans_en,trans_zh,project_dir,translations_dir
# p=re.compile(r'"(.*)"(=".*")')
text = '"result_training": "result of model training:",'

p = re.compile(r'\s*"(.+?)"(.*?:)(.*".+"),?')


r = "./en.json"
t = "./en.py"

r="./zh.json"
t="./zh.py"

r,t=translations_dir/r,translations_dir/t
with open(file=r,mode='r',encoding='utf-8') as fin:
    with open(file=t,mode='w',encoding='utf-8') as fout:
        for line in fin:
            if line.strip() in ['{','}']:
                continue
            res=p.sub(r"\1 = \3", line,1)
            fout.write(res)
            print(res)

# print(res,"@{res}")
# import en
# en.result_frame
