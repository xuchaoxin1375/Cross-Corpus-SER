# fileinput_cli.py
import fileinput
import sys
print(sys.argv)
for f in fileinput.input():
	print(f)
