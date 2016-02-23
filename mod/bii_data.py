# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 14:47:45 2016

@author: FO
"""

# Data

import pandas as pd


def bii_data(ident,code):
    
    df0 = pd.read_csv('fulldata.csv')
    
    if ident == 1:
        df0 = df0.loc[df0['Email Address'] == code]
    elif ident ==2:
        df0 = df0.loc[df0['Code'] == code]
        
    
    trust=df0['trust']=(.27*df0['QT1'].fillna(value=3)+.16*(6-df0['QT2'].fillna(value=3))+.34*df0['QT3'].fillna(value=3)+.22*(6-df0['QT1'].fillna(value=3)))* 9/4 - 5/4
    res=df0['res']=(.26*df0['QF1'].fillna(value=3)+.21*df0['QF2'].fillna(value=3)+.35*df0['QF3'].fillna(value=3)+.17*df0['QF4'].fillna(value=3))* 9/4 - 5/4
    div=df0['div']= (.15*df0['QD1'].fillna(value=3)+.29*df0['QD2'].fillna(value=3)+.35*df0['QD3'].fillna(value=3)+.20*df0['QD4'].fillna(value=3))* 9/4 - 5/4
    bel=df0['belief']=(.24*df0['QB1'].fillna(value=3)+.3*df0['QB2'].fillna(value=3)+.3*df0['QB3'].fillna(value=3)+.16*df0['QB4'].fillna(value=3))* 9/4 - 5/4
    collab=df0['collaboration']= (0*df0['QC1'].fillna(value=3)+ 0*df0['QC2'].fillna(value=3)+.59*df0['QC3'].fillna(value=3)+.41*df0['QC4'].fillna(value=3))* 9/4 - 5/4
    resall=df0['perfection']= (.55*(6-df0['QP3'].fillna(value=3))+.45*df0['QP4'].fillna(value=3))* 9/4 - 5/4
    czx=df0['CZX']= (df0['CZ'].fillna(value=2.5)-1)*3+1
    comfort=df0['CZ'].fillna(value=2.5)
    iz=df0['IZ']= (.47*df0['CZX']/2 + .53*df0['SDR'].fillna(value=3)) * 9/4- 5/4;  
    
    #Calculate overall score
    score=df0['score']=(df0['trust']+df0['res']+df0['div']+df0['belief']+df0['collaboration']+df0['perfection']+df0['IZ'])/7
    
    return [trust,res,div,bel,collab,resall,czx,comfort,iz,score]


# The data that is returned is returned in type pd.directory

#Check labels in data fram



