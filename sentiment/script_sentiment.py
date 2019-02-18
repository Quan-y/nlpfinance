#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Research on brexit data: sentiment analysis
@author: ericyuan
"""
from sentiment_api import Sentvisual, Textblob
import pandas as pd
import matplotlib.pyplot as plt

params = {'legend.fontsize': 12,
                      'figure.figsize': (12, 6),
                      'axes.labelsize': 18,
                      'axes.titlesize': 18,
                      'xtick.labelsize': 12,
                      'ytick.labelsize': 12,
                      'font.family': 'Times New Roman'}
plt.rcParams.update(params)


cleaned_data = pd.read_csv('../result/brexitText_cleaned.csv')
cleaned_data = cleaned_data.fillna("")

Visual = Sentvisual()
Textsent = Textblob()

bodyResult = Textsent.fit(cleaned_data['body'])
headResult = Textsent.fit(cleaned_data['headline'])

bodyResult.plot(title = "Body polar and subj", subplots = True)
headResult.plot(title = "Head polar and subj", subplots = True)

Visual.pie(bodyResult)
Visual.pie(headResult)

#bodyResult.to_csv('bodysentiment.csv')
#headResult.to_csv('headsentiment.csv')