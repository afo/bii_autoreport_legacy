# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 03:32:39 2015

@author: ikhlaqsidhu
"""

# General syntax to import specific functions in a library: 
##from (library) import (specific library function)

# General syntax to import a library but no functions: 
##import (library) as (give the library a nickname/alias)
import matplotlib.pyplot as plt
import pandas as pd #this is how I usually import pandas
import sys #only needed to determine Python version number
from pylab import *
import os
print os.getcwd()


# from StringIO import StringIO
# import pandas

df0 = pd.read_csv('fulldata.csv')

df0['trust']=(.27*df0['QT1'].fillna(value=3)+.16*(6-df0['QT2'].fillna(value=3))+.34*df0['QT3'].fillna(value=3)+.22*(6-df0['QT1'].fillna(value=3)))* 9/4 - 5/4
df0['res']=(.26*df0['QF1'].fillna(value=3)+.21*df0['QF2'].fillna(value=3)+.35*df0['QF3'].fillna(value=3)+.17*df0['QF4'].fillna(value=3))* 9/4 - 5/4
df0['div']= (.15*df0['QD1'].fillna(value=3)+.29*df0['QD2'].fillna(value=3)+.35*df0['QD3'].fillna(value=3)+.20*df0['QD4'].fillna(value=3))* 9/4 - 5/4
df0['belief']=(.24*df0['QB1'].fillna(value=3)+.3*df0['QB2'].fillna(value=3)+.3*df0['QB3'].fillna(value=3)+.16*df0['QB4'].fillna(value=3))* 9/4 - 5/4
df0['collaboration']= (0*df0['QC1'].fillna(value=3)+ 0*df0['QC2'].fillna(value=3)+.59*df0['QC3'].fillna(value=3)+.41*df0['QC4'].fillna(value=3))* 9/4 - 5/4
df0['perfection']= (.55*(6-df0['QP3'].fillna(value=3))+.45*df0['QP4'].fillna(value=3))* 9/4 - 5/4
df0['CZX']= (df0['CZ'].fillna(value=2.5)-1)*3+1
df0['IZ']= (.47*df0['CZX']/2 + .53*df0['SDR'].fillna(value=3)) * 9/4- 5/4;  

df0['score']=(df0['trust']+df0['res']+df0['div']+df0['belief']+df0['collaboration']+df0['perfection']+df0['IZ'])/7

#print df0.iloc[1]['Age']
print df0
df0.to_csv('foo.csv')


# print df0[1]
print df0[df0.PROJECT=='ARN16S']
df0[df0.PROJECT=='ARN16S'].to_csv('foo-filtered.csv')

plt.hist(df0['score'])

plt.show()

plt.plot([1.6, 2.7])

plt.show()
# df_gvl=df0[df0.Code=='GVL2016']
# print df_gvl
