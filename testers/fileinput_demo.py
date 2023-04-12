# fileinput_demo.py
import fileinput

with fileinput.input(files=('f1', 'f2')) as f:
    # python3.10后支持encoding等参数
    for line in f:
        # process(line)
        print(line)
