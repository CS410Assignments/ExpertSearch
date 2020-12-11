import os
import glob
import pandas as pd

#Merge input txt files into one file

filenames = ["UTexas1.txt", "UTexas2.txt", "UTexas3.txt", "UTexas4.txt", "UTexas5.txt"]

with open("UTexas.txt", "w") as outfile:
    i=0
    for filename in filenames:
        if i != 0:
            outfile.write('\n')
        i = i + 1
        with open(filename) as infile:
            contents = infile.read()
            outfile.write(contents)
