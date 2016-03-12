import numpy as np
import pandas as pd
from pylab import *
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def bii_vbar(ident,code,in_data):
    if ident != 2:
        i_string = ident
    else:
        plt.figure()
        i_string = code
        
        [trust,res,div,bel,collab,resall,czx,comfort,iz,score] = in_data
        
        N = 8
        values = [mean(trust),mean(res),mean(div),mean(bel),mean(collab),mean(resall),mean(czx),mean(iz)]
        valStd = [std(trust),std(res),std(div),std(bel),std(collab),std(resall),std(czx),std(iz)]
        
        
        ind = np.arange(N)  # the x locations for the groups
        width = 0.35       # the width of the bars
        
        fig, ax = plt.subplots()
        rects = ax.bar(ind, values, width, color='b', yerr=valStd)
        
        # add some text for labels, title and axes ticks
        ax.set_ylabel('Score')
        #ax.set_title('Results for ' + i_string,loc='left')
        ax.set_xticks(ind + width)
        ax.set_xticklabels(('Trust', 'Res', 'Div', 'Ment Str','Res All', 'Collab', 'Com Zone', 'In Zone'))
        ax.set_ylim([0,10])
        
        
        ax.legend((rects[0]), ('Individual', 'Mean'),bbox_to_anchor=(1.3, 1.3),borderaxespad=0.)
        
        
        def autolabel(rects):
            # attach some text labels
            for rect in rects:
                height = float(rect.get_height())
                ax.text(float(rect.get_x()) + rect.get_width()/2.0, 1.02*float(height),
                        round(height,1),
                        ha='center', va='bottom',fontsize=9)
        
        autolabel(rects)
        plt.figtext(0.52, 1.02, 'Results for '+ i_string,
        ha='center', color='black', size='large',fontweight='bold')
        plt.xlabel(r'$\mathrm{Innovation \ Index \ Score:}\ %.3f$' %(mean(score)),fontsize='18', fontweight='bold')
        plt.show()