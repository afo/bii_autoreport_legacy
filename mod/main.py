# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 14:27:08 2016

@author: alexanderfo & Johan
"""

# Universal import
import os
import shutil
import pandas as pd
from sys import argv
from pylab import *
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
import matplotlib.mlab as mlab

from flask import Flask
from flask import render_template

# Local import
from bii_data import bii_data
from bii_plot.bii_radar import *
from bii_plot.bii_hist import *
from bii_plot.bii_hbar import *


# Add comparison variable
#comparison = 'JMU-JS16' # comparison is wg-code or email
# Update when added comparison


if len(sys.argv) == 2:
    mail = argv[1] #mail
    wg = None
elif len(sys.argv) == 3:
    mail = argv[1] #mail
    wg = argv[2] #wg-code

# Identify type
if wg is None:
    ident = 1 # individual
    data = bii_data(ident,mail)

        
else:
    ident = 2 #workgroup
    data= bii_data(ident,wg)

[trust,res,div,bel,collab,resall,czx,comfort,iz,score] = data


#Plotting

# Just for dev...
#if os.path.exists('/Users/johanenglarsson/bii/mod/static'):
 #   shutil.rmtree('/Users/johanenglarsson/bii/mod/static')

# Check if static-directory is created
if not os.path.exists('/Users/johanenglarsson/bii/mod/static'):
    os.makedirs('/Users/johanenglarsson/bii/mod/static')

bii_radar(ident, wg, data)
bii_hist(ident,wg,score,'Innovation Index Score', 'green',1)
bii_hist(ident,wg,comfort,'Comfort Zone Score', 'yellow',2)
bii_hbar(ident, wg, data)

#This should remove a folder...don't think it is necessary
#shutil.rmtree('/Users/johanenglarsson/bii/mod/static')




# Send images to HTML-template

app = Flask(__name__)

@app.route('/BII/<wg>')
def hello(wg=None):
    return render_template('report.html', wg = wg, index_score=round(mean(score),2), tru = round(mean(trust),2), col = round(mean(collab),2), res_all = round(mean(resall),2), div = round(mean(div),2), men_st = round(mean(bel),2), in_zone = round(mean(iz),2), res = round(mean(res),2))
    
app.run()






#if comparison != None:
 #   if '@' in comparison:
  #      ident=1
   #     comp_data = bii_data(ident,comparison)
    #else:
     #   ident=2
      #  comp_data = bii_data(ident,comparison)


#if Comparison:
 #   ident = 3
  #  data_comp = bii_data(ident, Comparison)
    









def main():
    print type


if (__name__ == "__main__"):
    main()