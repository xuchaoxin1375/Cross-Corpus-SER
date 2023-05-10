##
import re
p=re.compile(r'\s*(.*)(\s*=.*)')
from config.MetaPath import trans_en,trans_zh,project_dir,translations_dir

res=[]
with open(trans_en,'r',encoding='utf-8')as fe:
    with open(trans_zh,'r',encoding='utf-8')as fz:
        i=1
        for e,z in zip(fe,fz):
            # print(e,z)
            # print(i)
            # i+=1
            if e.strip() and z.strip():
                varName=p.search(e).group(1)
                # print('varName: ', varName)
                value=p.search(z).group(2)
                line=f'{varName}{value}'
                # print(line)
                res.append(line)
            else:res.append(z.strip())
with open(trans_zh,'w',encoding='utf-8') as fout:
    for line in res:
        fout.write(line+'\n')
        print(line)