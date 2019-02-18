#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sentiment analysis
@author: ericyuan
"""
import pandas as pd
# Textblob
from textblob import TextBlob
# Visualization
import matplotlib.pyplot as plt

class Sentvisual:
    '''
    Input: dataframe with column name "polar", "subj"
    Output: graphs
    '''
    def setting(self, **params):
        if params is None:
            params = {'legend.fontsize': 12,
                      'figure.figsize': (12, 6),
                      'axes.labelsize': 18,
                      'axes.titlesize': 18,
                      'xtick.labelsize': 12,
                      'ytick.labelsize': 12,
                      'font.family': 'Times New Roman'}
        plt.rcParams.update(params)
    
    def line(self, sentimentDf):
        sentimentDf.plot(title = "polar and subj", subplots = True)
        
    def pie(self, sentimentDf, posSent = 0, negSent = 0, \
            posSub = 0.5, negSub = 0.5):
        '''
        posSent, negSent, posSub, negSub are both threshold
        '''
        # pie for sentiment
        labelsSent = ['positive', 'negative']
        sentSizes = [len(sentimentDf[sentimentDf['polar'] >= posSent]), \
                len(sentimentDf[sentimentDf['polar'] < negSent])]
        plt.pie(sentSizes, labels=labelsSent, autopct='%1.1f%%')
        plt.show()
        # pie for subjectivity
        labelsSub = ['subjective', 'objective']
        sizes = [len(sentimentDf[sentimentDf['subj'] >= posSub]), \
                len(sentimentDf[sentimentDf['polar'] < negSub])]
        plt.pie(sizes, labels=labelsSub, autopct='%1.1f%%')
        plt.show()

class Textblob:
    def __mapfun(self, text):
        '''
        text: string
        '''
        text = TextBlob(text)
        result = text.sentiment
        return (result[0], result[1])
    
    def fit(self, textSeries):
        '''
        Input: list/tuple/pandas.Series [sentence, sentence, ...]
        Output: dataframe with column name "polar", "subj"
        
        Explanation of output:
            (1) The polarity score is a float within the range [-1.0, 1.0]
            where -1.0 is very negative and 1.0 is very positive
            (2) The subjectivity is a float within the range [0.0, 1.0] 
            where 0.0 is very objective and 1.0 is very subjective.
        '''
        txt = pd.Series(textSeries)
        txt = txt.map(self.__mapfun)
        txt = txt.apply(pd.Series)
        txt.columns = ['polar', 'subj']
        return txt



