import numpy as np
import pandas as pd
from pylab import *
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def bii_hbar(ident,code,in_data):
    [trust,res,div,bel,collab,resall,czx,comfort,iz,score] = in_data
    plt.figure()
    if ident != 2:
        ident = ident
    else:

        if ident != 2:
            i_string = ident
        else:
            i_string = code
            
        
        val = [mean(trust),mean(res),mean(div),mean(bel),mean(collab),mean(resall),mean(czx),mean(iz)][-1::-1]
        err = [std(trust),std(res),std(div),std(bel),std(collab),std(resall),std(czx),std(iz)][-1::-1]
        pos = arange(8)    # the bar centers on the y axis


        
        
        
        
        plt.plot((mean(score), mean(score)), (-1, 8), 'g',label='Average',linewidth=3)
        plt.barh(pos,val, xerr=err, ecolor='r', align='center',label='Score')
        plt.errorbar(val,pos, xerr=err, label="St Dev", color='r',fmt='o')
        plt.legend(loc='upper center', shadow=True, fontsize='x-large',bbox_to_anchor=(1.1, 1.1),borderaxespad=0.)
        plt.yticks(pos, (('Trust', 'Resilience', 'Diversity', 'Confidence','Resource Allocation', 'Collaboration', 'Comfort Zone', 'Innovation Zone'))[-1::-1])
        plt.xlabel('Score')
        plt.title('Results for ' + i_string, fontweight='bold', y=1.01)
        plt.xlabel(r'$\mathrm{Total \ Innovation \ Index \ Score:}\ %.3f$' %(mean(score)),fontsize='18')
        axes = plt.gca()
        axes.set_xlim([0,10])
#        plt.legend((score_all,score_mean), ('Score','Mean'),bbox_to_anchor=(1.3, 1.3),borderaxespad=0.)
        file_name = "hbar"
        path_name = "/Users/johanenglarsson/bii/mod/static/%s" %file_name
        plt.savefig(path_name)