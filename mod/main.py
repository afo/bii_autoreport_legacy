# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 14:27:08 2016

@author: alexanderfo & Johan
"""

# Universal import
import matplotlib.pyplot as plt
import pandas as pd
from sys import argv
from pylab import *
import numpy as np
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from scipy.stats import norm
import matplotlib.mlab as mlab

# Local import
from bii_data import bii_data

from bii_plot.radar1 import bii_radar1
from bii_plot.radar2 import bii_radar2


# Add comparison variable
Comparison = 'JMU-JS16' # comparison is wg-code or email
# Update when added comparison
if len(sys.argv) == 2:
    mail = argv[1] #mail
    wg = None
elif len(sys.argv) == 
    mail = argv[1] #mail
    wg = argv[2] #wg-code

# Identify type
if wg is None:
    ident = 1 # individual
    data = bii_data(ident,mail)
        
        
    
    
    
    
else:
    ident = 2 #workgroup
    data= bii_data(ident,wg)
    



if comparison != None:
    if '@' in comparison:
        ident=1
        comp_data = bii_data(ident,comparison)
    else:
        ident=2
        comp_data = bii_data(ident,comparison)
    
    return comp_data


if Comparison:
    ident = 3
    data_comp = bii_data(ident, Comparison)
    









def main():
    print type


if (__name__ == "__main__"):
    main()