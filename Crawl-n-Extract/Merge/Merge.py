import os
import glob
import pandas as pd

filenames = ["UIUC1.txt", "UIUC2.txt"]

with open("UIUC.txt", "w") as outfile:
    i=0
    for filename in filenames:
        if i != 0:
            outfile.write('\n')
        i = i + 1
        with open(filename) as infile:
            contents = infile.read()
            outfile.write(contents)
