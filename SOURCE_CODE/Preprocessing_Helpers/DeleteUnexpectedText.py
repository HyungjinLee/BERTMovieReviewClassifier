#!/usr/bin/env python
# coding: utf-8

import csv
import os
import pandas as pd

#!/usr/bin/env python3
# Removing an Unexpected Text which prohibits my code from operating properly.

import fileinput

with fileinput.FileInput('../Data/KaggleTestData.txt', inplace=True, backup='.bak') as file:
    for line in file:
        print(line.replace('', ''), end='')