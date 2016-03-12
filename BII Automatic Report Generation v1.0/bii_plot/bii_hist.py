import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from pylab import *
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# read data from a text file. One number per line

def bii_hist(ident,code,hdat,hstr,col,g_type):
    if ident != 2:
    	ident = ident
    else:          


        if len(code) == 2 and not isinstance(code, basestring):
            code = code[0] + " " + code[1]

        datos=hdat
        plt.figure()
        
        # best fit of data
        (mu, sigma) = norm.fit(datos)
        no_bins = 10
        if len(set(hdat)) < 20:
            no_bins = len(set(hdat))
        
        # the histogram of the data
        n, bins, patches = plt.hist(datos.values, no_bins, facecolor=col, alpha=0.75,label='Counts')
        
        height = n.max()
        
        # add a 'best fit' line
        y = mlab.normpdf( bins, mu, sigma)
        l = plt.plot(bins, y*height, 'r--', linewidth=1,label='Distribution')
        
        #plot
        

        plt.title(r'$\mathrm{Workgroup \ Result \ Statistics \ :}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma),fontsize=18,y=-0.29)
        plt.ylabel('Counts',fontsize=12)
        plt.xlabel(hstr)
        tit = plt.suptitle('Histogram of ' + hstr + ' ' + code, fontsize=14, fontweight='bold', y=1.05)
        plt.grid(True)
        plt.plot((mu, mu), (0, height), 'b',linewidth=2,label='Average')
        lgd = plt.legend(loc='upper center', shadow=True, fontsize='x-large',bbox_to_anchor=(1.2, 1),borderaxespad=0.)
        
        if g_type==1:
        	file_name = "hist_score"
        else:
        	file_name = "hist_cz"
        path_name = "static/%s" %file_name
    	#path_name = "/Users/johanenglarsson/bii/mod/static/%s" %file_name
        plt.tight_layout()
    	plt.savefig(path_name, bbox_extra_artists=(lgd,tit,), bbox_inches='tight')


