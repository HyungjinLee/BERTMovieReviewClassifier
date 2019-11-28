#!/usr/bin/env python
# coding: utf-8

import csv
import os
import pandas as pd

f = open('/../Data/rt-polarity-prev.pos')

with open('/../Data/output.txt', 'a') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t') 
    tsv_writer.writerow(['text', 'sentiment'])
    
    # For all lines in a recent file
    
    while True:
        line = f.readline()
        if not line: break
        line = line.strip('\n')
        #print(line.strip('\n'))
        
        tsv_writer.writerow([line,'1'])
    f.close()

df = pd.read_csv('/../Data/output.txt', sep='\t')

df.shape

df.tail()