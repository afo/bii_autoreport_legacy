import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from pylab import *
from scipy.stats import norm
import matplotlib

matplotlib.use('Agg')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def bii_hbar(group,code,in_data):
    [trust,res,div,bel,collab,resall,comfort,iz,score] = in_data
    plt.figure()

    if len(code) == 2 and not isinstance(code, basestring):
        code = code[0] + " " + code[1]
            
        
    val = [mean(trust),mean(res),mean(div),mean(bel),mean(collab),mean(resall),mean(comfort),mean(iz)][-1::-1]
    pos = arange(8)    # the bar centers on the y axis


        
        
        
        
    plt.plot((mean(score), mean(score)), (-1, 8), 'g',label='Average',linewidth=3)
    #plt.barh(pos,val, xerr=err, ecolor='r', align='center',label='Score')
    plt.barh(pos,val, align='center', label='Score')
    if group:
        err = [std(trust),std(res),std(div),std(bel),std(collab),std(resall),std(comfort),std(iz)][-1::-1]
        plt.errorbar(val,pos, xerr=err, label="St Dev", color='r',fmt='o')

    lgd = plt.legend(loc='upper center', shadow=True, fontsize='x-large',bbox_to_anchor=(1.1, 1.1),borderaxespad=0.)
    plt.yticks(pos, (('Tru', 'Res', 'Div', 'Ment Str','Collab', 'Res All', 'Com Zone', 'In Zone'))[-1::-1])
    plt.xlabel('Score')
    plt.title('Results for ' + code, fontweight='bold', y=1.01)
    plt.xlabel(r'$\mathrm{Total \ Innovation \ Index \ Score:}\ %.3f$' %(mean(score)),fontsize='18')
    axes = plt.gca()
    axes.set_xlim([0,10])
#        plt.legend((score_all,score_mean), ('Score','Mean'),bbox_to_anchor=(1.3, 1.3),borderaxespad=0.)
    file_name = "hbar"
    path_name = "static/%s" %file_name
        #path_name = "/Users/johanenglarsson/bii/mod/static/%s" %file_name
    plt.savefig(path_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
