# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 13:06:41 2016

@author: FO
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:20:05 2016

@author: FO
"""



import matplotlib.pyplot as plt
import pandas as pd
from pylab import *
import os
print os.getcwd()


df0 = pd.read_csv('fulldata.csv')
trust_tot=df0['trust']=(.27*df0['QT1'].fillna(value=3)+.16*(6-df0['QT2'].fillna(value=3))+.34*df0['QT3'].fillna(value=3)+.22*(6-df0['QT1'].fillna(value=3)))* 9/4 - 5/4
res_tot=df0['res']=(.26*df0['QF1'].fillna(value=3)+.21*df0['QF2'].fillna(value=3)+.35*df0['QF3'].fillna(value=3)+.17*df0['QF4'].fillna(value=3))* 9/4 - 5/4
div_tot=df0['div']= (.15*df0['QD1'].fillna(value=3)+.29*df0['QD2'].fillna(value=3)+.35*df0['QD3'].fillna(value=3)+.20*df0['QD4'].fillna(value=3))* 9/4 - 5/4
bel_tot=df0['belief']=(.24*df0['QB1'].fillna(value=3)+.3*df0['QB2'].fillna(value=3)+.3*df0['QB3'].fillna(value=3)+.16*df0['QB4'].fillna(value=3))* 9/4 - 5/4
collab_tot=df0['collaboration']= (0*df0['QC1'].fillna(value=3)+ 0*df0['QC2'].fillna(value=3)+.59*df0['QC3'].fillna(value=3)+.41*df0['QC4'].fillna(value=3))* 9/4 - 5/4
perf_tot=df0['perfection']= (.55*(6-df0['QP3'].fillna(value=3))+.45*df0['QP4'].fillna(value=3))* 9/4 - 5/4
czx_tot=df0['CZX']= (df0['CZ'].fillna(value=2.5)-1)*3+1
iz_tot=df0['IZ']= (.47*df0['CZX']/2 + .53*df0['SDR'].fillna(value=3)) * 9/4- 5/4; 
score_tot=df0['score']=(df0['trust']+df0['res']+df0['div']+df0['belief']+df0['collaboration']+df0['perfection']+df0['IZ'])/7












wg=False
i='andrew_k_yu@yahoo.com'
i='yadavsuren@gmail.com'


i=False
wg='ELPP-2015F'
wg='BM-BC-2016S'





bii_data(i,wg)

bii_hbar1(i,wg)












### READ IN DATA and Specify data variables


def bii_data(i,wg):
    
    df0 = pd.read_csv('fulldata.csv')
    
    if i != False:
        df0 = df0.loc[df0['Email Address'] == i]
    elif wg != False:
        df0 = df0.loc[df0['Code'] == wg]
    else:
        df0 = df0
        
    global trust
    global res
    global div
    global bel
    global collab
    global perf
    global czx
    global iz
    global score
    
    trust=df0['trust']=(.27*df0['QT1'].fillna(value=3)+.16*(6-df0['QT2'].fillna(value=3))+.34*df0['QT3'].fillna(value=3)+.22*(6-df0['QT1'].fillna(value=3)))* 9/4 - 5/4
    res=df0['res']=(.26*df0['QF1'].fillna(value=3)+.21*df0['QF2'].fillna(value=3)+.35*df0['QF3'].fillna(value=3)+.17*df0['QF4'].fillna(value=3))* 9/4 - 5/4
    div=df0['div']= (.15*df0['QD1'].fillna(value=3)+.29*df0['QD2'].fillna(value=3)+.35*df0['QD3'].fillna(value=3)+.20*df0['QD4'].fillna(value=3))* 9/4 - 5/4
    bel=df0['belief']=(.24*df0['QB1'].fillna(value=3)+.3*df0['QB2'].fillna(value=3)+.3*df0['QB3'].fillna(value=3)+.16*df0['QB4'].fillna(value=3))* 9/4 - 5/4
    collab=df0['collaboration']= (0*df0['QC1'].fillna(value=3)+ 0*df0['QC2'].fillna(value=3)+.59*df0['QC3'].fillna(value=3)+.41*df0['QC4'].fillna(value=3))* 9/4 - 5/4
    perf=df0['perfection']= (.55*(6-df0['QP3'].fillna(value=3))+.45*df0['QP4'].fillna(value=3))* 9/4 - 5/4
    czx=df0['CZX']= (df0['CZ'].fillna(value=2.5)-1)*3+1
    iz=df0['IZ']= (.47*df0['CZX']/2 + .53*df0['SDR'].fillna(value=3)) * 9/4- 5/4;  
    
    #Calculate overall score
    score=df0['score']=(df0['trust']+df0['res']+df0['div']+df0['belief']+df0['collaboration']+df0['perfection']+df0['IZ'])/7

#Check labels in data fram







### Lying barchart 1
from pylab import *


#        
#        plt.show()





#        
#        # Example data
#        titles = ('Trust', 'Res', 'Div', 'MS','Perf', 'Collab', 'RA', 'IZ')
#        y_pos = np.arange(len(titles))
#        performance = [mean(trust),mean(res),mean(div),mean(bel),mean(collab),mean(perf),mean(czx),mean(iz)]
#        error = [std(trust),std(res),std(div),std(bel),std(collab),std(perf),std(czx),std(iz)]
#        frameon=False
#        plt.barh(y_pos, performance, xerr=error, align='center', alpha=0.4)
#        plt.yticks(y_pos, titles)
#        plt.title('Score for ' + i_string)
#        plt.xlabel(r'$\mathrm{Innovation \ Index \ Score:}\ %.3f$' %(mean(score)),fontsize='18', fontweight='bold')
#        axes = plt.gca()
#        axes.set_xlim([0,10])
#        plt.plot((mean(score), mean(score)), (-1, 8), 'r')
#        plt.show()
    
    