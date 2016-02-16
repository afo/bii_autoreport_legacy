# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 13:59:26 2016

@author: jean-etiennegoubet
"""

#try with BII
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
from pylab import *
import os
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.stats import norm


### READ IN DATA and Specify data variables

df0 = pd.read_csv('/Users/jean-etiennegoubet/Documents/ENS/Berkeley/bii/fulldata.csv')
trust=df0['trust']=(.27*df0['QT1'].fillna(value=3)+.16*(6-df0['QT2'].fillna(value=3))+.34*df0['QT3'].fillna(value=3)+.22*(6-df0['QT1'].fillna(value=3)))* 9/4 - 5/4
res=df0['res']=(.26*df0['QF1'].fillna(value=3)+.21*df0['QF2'].fillna(value=3)+.35*df0['QF3'].fillna(value=3)+.17*df0['QF4'].fillna(value=3))* 9/4 - 5/4
div=df0['div']= (.15*df0['QD1'].fillna(value=3)+.29*df0['QD2'].fillna(value=3)+.35*df0['QD3'].fillna(value=3)+.20*df0['QD4'].fillna(value=3))* 9/4 - 5/4
bel=df0['belief']=(.24*df0['QB1'].fillna(value=3)+.3*df0['QB2'].fillna(value=3)+.3*df0['QB3'].fillna(value=3)+.16*df0['QB4'].fillna(value=3))* 9/4 - 5/4
collab=df0['collaboration']= (0*df0['QC1'].fillna(value=3)+ 0*df0['QC2'].fillna(value=3)+.59*df0['QC3'].fillna(value=3)+.41*df0['QC4'].fillna(value=3))* 9/4 - 5/4
perf=df0['perfection']= (.55*(6-df0['QP3'].fillna(value=3))+.45*df0['QP4'].fillna(value=3))* 9/4 - 5/4
czx=df0['CZX']= (df0['CZ'].fillna(value=2.5)-1)*3+1
iz=df0['IZ']= (.47*df0['CZX']/2 + .53*df0['SDR'].fillna(value=3)) * 9/4- 5/4;  

#Sort the data

fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed

pl.plot(h,fit,'-o')

pl.hist(h,normed=True)      #use this to draw histogram of your data

#Define variables for graph

trace1 = go.Histogram(
    y=trust,
    opacity=0.75
)
trace2 = go.Histogram(
    y=res,
    opacity=0.75
)
trace3 = go.Histogram(
    y=div,
    opacity=0.75
)
trace4 = go.Histogram(
    y=bel,
    opacity=0.75
)
trace5 = go.Histogram(
    y=collab,
    opacity=0.75
)
trace6 = go.Histogram(
    y=perf,
    opacity=0.75
)
trace7 = go.Histogram(
    y=czx,
    opacity=0.75
)
trace8 = go.Histogram(
    y=iz,
    opacity=0.75
)
data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8]
layout = go.Layout(
    barmode='horizontal'
)
fig = go.Figure(data=data, layout=layout)
plot_url = py.plot(fig, filename='horizontal-histogram')
