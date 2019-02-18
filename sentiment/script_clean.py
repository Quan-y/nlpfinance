#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Research on brexit data: cleaning
@author: ericyuan
"""
import pandas as pd
from datetime import datetime
import re
# ----------------------------------------------------------------------- #
# ----------------------- Clean NaN and time type ----------------------- #
# ----------------------------------------------------------------------- #
# load data
textColumns = ['altId', 'firstCreated', 'body', 'headline']
brexitText = pd.read_csv('../data/Brexit.csv', index_col = False, \
                     usecols = textColumns)
brexitText.columns = ['id', 'body', 'time', 'headline']
print("Length of data {0}".format(len(brexitText)))

# remove rows if both 'body' and 'headline' are NaN (4 rows)
brexitText = brexitText[(brexitText['body'] == brexitText['body'])\
                        |(brexitText['headline'] == brexitText['headline'])]

# remove rows if time is NaN (1 row)
brexitText = brexitText[~brexitText['time'].isna()]
print("Length of data {0} after removing NaN".format(len(brexitText)))

# remove rows the length of time string != 24 (1 row)
brexitText = brexitText[brexitText['time'].map(len)==24]

# count the number of NaN
print()
for each_col in brexitText.columns:
    print("{0} has {1} NaN value".format(each_col, sum(brexitText[each_col].isna())))

# convert type of time in firstCreated column
convertTime = lambda x: datetime.strptime(x[:-5], '%Y-%m-%dT%H:%M:%S')
brexitText['time'] = brexitText['time'].map(convertTime)

# ----------------------------------------------------------------------- #
# ----------------------------- Clean Text ------------------------------ #
# ----------------------------------------------------------------------- #
# After reviewing 300 rows of news body
# 1. remove urls
reg1 = r"\S*(www\.[^:\/\n]+\.com)\s*"
reg11 = r"https?:.+"
# 2. remove email address
reg2 = r"\S+@\S+"
# 3. remove [XXXXXXX] length = 12
reg3 = r"\[.{0,12}\]"
# 4. remove ((XXXXXXX))
reg4 = r"\(\(.*\)\)"
# 5. remove <XXXXXXX> length <= 12
reg5 = r"\<.{0,12}\>"
# 6. remove signle char blank_space(single_char)blank_space 
reg6 = r"((?<=^)|(?<= )).((?=$)|(?= ))"
# 7. remove telephone number +44 207 542 3213
reg7 = r"\s[0-9]{8,}"
# 8. remove ...
reg8 = r"\.{3,}"
# 9. remove letter.whitespace
reg9 = r"[a-zA-Z].\s"
# 10. remove non English, number, -, $, ., %, ', /
reg10 = r"[^a-z|A-Z|0-9|_|\-|$|\.|%|\'|\/]"
# 12. for XXX.
reg12 = r"([a-zA-Z]+)\."

# for headline
# 1. remove non English, number, -, $, ., %, ', /
# 2. remove <XXXXXXX> length <= 10
# 3. remove signle char blank_space(single_char)blank_space 
regBody =  re.compile("(%s|%s|%s|%s|%s|%s|%s|%s|%s)" % \
                      (reg1, reg2, reg3, reg4, reg5, reg8, reg9, reg10, reg6), \
                      re.S)
regHead = re.compile("(%s|%s|%s)" % \
                      (reg5, reg10, reg6), \
                      re.S)
bodyregList = [reg4, reg3, reg5, reg11, reg1, reg2, reg7, reg8, reg10, reg6]
headregList = [reg5, reg10, reg6]
# define function
cleaned_brexitText = brexitText.copy()

def regclean(s, reglist):
    regx = r"([a-zA-Z]+)\."
    if s == s:
        for i in reglist:
            s = re.sub(i, ' ', s)
        s = re.sub(regx, r' \1', s)
    return s

# clean body
cleaned_brexitText['body'] = cleaned_brexitText['body'].map(lambda x: regclean(x, bodyregList))
# clean head
cleaned_brexitText['headline'] = cleaned_brexitText['headline'].map(lambda x: regclean(x, headregList))
#cleaned_brexitText.to_csv('brexitText_cleaned.csv')



