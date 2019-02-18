#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
text data cleaning
@author: ericyuan
"""
import re
import pandas as pd

class Regclean:
    def mapfun(self, x, regex):
            if type(x) == str:
                for each_pattern in regex:
                    pattern = re.compile(each_pattern, re.S)
                    x = re.sub(pattern, ' ', x)
                    return x
            else:
                return x
    def remove(self, text, regex):
        '''
        Input: 
        text: list/series/tuple
        regex: list of string (regular expression)
        
        Output:
        Series
        '''
        text = pd.Series(text)
        text = text.map(lambda x: self.mapfun(x, regex))
        return text
    def extrect(self, text, regex, numgp, namegp = None):
        '''
        Input:
        text: list/series/tuple
        regex: string (regular expression with groups)
        numgp: the number of groups in regular expression
        namegp(optional): the name of groups
        
        Output:
        DataFrame
        '''
        text = pd.Series(text)
        # map function
        def mapfun(s):
            result = re.match(regex, s)
            listresult = []
            for each_group in range(1, numgp+1):
                listresult.append(result.group(each_group))
            return listresult
        text = text.map(mapfun)
        text = text.apply(pd.Series)
        if namegp != None:
            text.columns = namegp
        return text






    