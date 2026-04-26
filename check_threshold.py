import sys

acc = float(open("accuracy.txt").read())

if acc < 0.85:
    sys.exit(1)
