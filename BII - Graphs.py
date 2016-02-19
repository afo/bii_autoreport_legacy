# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 13:59:26 2016

@author: jean-etiennegoubet
"""

# Setup

import os
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from scipy.stats import norm
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
from pylab import *

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

### Category data

## dataframe gender

df_men=df0.loc[df0['Gender'] == 2]
df_women=df0.loc[df0['Gender'] == 1]

# averages 

trust_av_men=mean(df_men['trust'])
trust_av_women=mean(df_women['trust'])

res_av_men=mean(df_men['res'])
res_av_women=mean(df_women['res'])

div_av_men=mean(df_men['div'])
div_av_women=mean(df_women['div'])

bel_av_men=mean(df_men['belief'])
bel_av_women=mean(df_women['belief'])

collab_av_men=mean(df_men['collaboration'])
collab_av_women=mean(df_women['collaboration'])

perf_av_men=mean(df_men['perfection'])
perf_av_women=mean(df_women['perfection'])

czx_av_men=mean(df_men['CZX'])
czx_av_women=mean(df_women['CZX'])

iz_av_men=mean(df_men['IZ'])
iz_av_women=mean(df_women['IZ'])

#quantity

# dataframe age

# dataframe study



## GRAPHS 

# Average data - Histogram

data = [
    go.Bar(
        x=['Trust', 'Resilience', 'Diversity', 'Belief', 'Collaboration', 'Perfection', 'CZX', 'IZ'],
        y=[mean(trust), mean(res), mean(div), mean(bel), mean(collab), mean(perf), mean(czx), mean(iz)]
    )
]
plot_url = py.plot(data, filename='basic-bar')


# Men vs Women -Bar charts

trace1 = go.Bar(
    x=['Trust', 'Resilience', 'Diversity', 'Belief', 'Collaboration', 'Perfection', 'CZX', 'IZ'],
    y=[trust_av_men, res_av_men, div_av_men, bel_av_men, collab_av_men, perf_av_men, czx_av_men, iz_av_men],
    name='Men'
)
trace2 = go.Bar(
    x=['Trust', 'Resilience', 'Diversity', 'Belief', 'Collaboration', 'Perfection', 'CZX', 'IZ'],
    y=[trust_av_women, res_av_women, div_av_women, bel_av_women, collab_av_women, perf_av_women, czx_av_women, iz_av_women],
    name='Women'
)
data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)
fig = go.Figure(data=data, layout=layout)
plot_url = py.plot(fig, filename='grouped-abr')


