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

df0 = pd.read_csv('fulldata.csv')
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


## dataframe age

df_age_young=df0.loc[df0['Age']== 1]
df_age_mid=df0.loc[df0['Age']== 2]
df_age_old=df0.loc[df0['Age']== 3]

young_av_trust=mean(df_age_young['trust'])
mid_av_trust=mean(df_age_young['trust'])
old_av_trust= mean(df_age_young['trust'])

young_av_res=mean(df_age_young['res'])
mid_av_res=mean(df_age_young['res'])
old_av_res= mean(df_age_young['res'])

young_av_div=mean(df_age_young['div'])
mid_av_div=mean(df_age_young['div'])
old_av_div= mean(df_age_young['div'])

young_av_bel=mean(df_age_young['belief'])
mid_av_bel=mean(df_age_young['belief'])
old_av_bel= mean(df_age_young['belief'])

young_av_collab=mean(df_age_young['collaboration'])
mid_av_collab=mean(df_age_young['collaboration'])
old_av_collab= mean(df_age_young['collaboration'])

young_av_perf=mean(df_age_young['perfection'])
mid_av_perf=mean(df_age_young['perfection'])
old_av_perf= mean(df_age_young['perfection'])

young_av_czx=mean(df_age_young['CZX'])
mid_av_czx=mean(df_age_young['CZX'])
old_av_czx= mean(df_age_young['CZX'])

young_av_iz=mean(df_age_young['IZ'])
mid_av_iz=mean(df_age_young['IZ'])
old_av_iz= mean(df_age_young['IZ'])

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


# Men vs Women - Bar chart

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
plot_url = py.plot(fig, filename='grouped-abr',show_link=False,auto_open=True)


# Age distribution - Stacked bar chart

trace1 = go.Bar(
    x=['Trust', 'Resilience', 'Diversity', 'Belief', 'Collaboration', 'Perfection', 'CZX', 'IZ'],
    y=[young_av_trust, young_av_res, young_av_div, young_av_bel, young_av_collab, young_av_perf, young_av_czx, young_av_iz],
    name='Young'
)
trace2 = go.Bar(
    x=['Trust', 'Resilience', 'Diversity', 'Belief', 'Collaboration', 'Perfection', 'CZX', 'IZ'],
    y=[mid_av_trust, mid_av_res, mid_av_div, mid_av_bel, mid_av_collab, mid_av_perf, mid_av_czx, mid_av_iz],
    name='Mid'
)
trace3 = go.Bar(
    x=['Trust', 'Resilience', 'Diversity', 'Belief', 'Collaboration', 'Perfection', 'CZX', 'IZ'],
    y=[old_av_trust, old_av_res, old_av_div, old_av_bel, old_av_collab, old_av_perf, old_av_czx, old_av_iz],
    name='Old'
)
data = [trace1, trace2, trace3]
layout = go.Layout(
    barmode='stack'
)
fig = go.Figure(data=data, layout=layout)
plot_url = py.plot(fig, filename='stacked-bar')