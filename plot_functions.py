# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:20:05 2016

@author: Alexander Fred Ojala
"""



import matplotlib.pyplot as plt
import pandas as pd
from pylab import *
import os
print os.getcwd()
from sys import argv

individual = argv[1]
workgroup = argv[2]
ind_in_workgroup = argv[3]


def bii_data(i,wg):
    
    df = pd.read_csv('fulldata.csv')
    
    if i != False:
        df0 = df.loc[df['Email Address'] == i]
    elif wg != False:
        df0 = df.loc[df0['Code'] == wg]
    elif wg == 'Total':
        wg = 'Total'
        
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





def bii_group_data(code):
    global trust_tot
    global res_tot
    global div_tot
    global bel_tot
    global collab_tot
    global perf_tot
    global czx_tot
    global iz_tot
    global score_tot
    df_code = pd.read_csv('fulldata.csv')
    trust_tot=df_code['trust']=(.27*df_code['QT1'].fillna(value=3)+.16*(6-df_code['QT2'].fillna(value=3))+.34*df_code['QT3'].fillna(value=3)+.22*(6-df_code['QT1'].fillna(value=3)))* 9/4 - 5/4
    res_tot=df_code['res']=(.26*df_code['QF1'].fillna(value=3)+.21*df_code['QF2'].fillna(value=3)+.35*df_code['QF3'].fillna(value=3)+.17*df_code['QF4'].fillna(value=3))* 9/4 - 5/4
    div_tot=df_code['div']= (.15*df_code['QD1'].fillna(value=3)+.29*df_code['QD2'].fillna(value=3)+.35*df_code['QD3'].fillna(value=3)+.20*df_code['QD4'].fillna(value=3))* 9/4 - 5/4
    bel_tot=df_code['belief']=(.24*df_code['QB1'].fillna(value=3)+.3*df_code['QB2'].fillna(value=3)+.3*df_code['QB3'].fillna(value=3)+.16*df_code['QB4'].fillna(value=3))* 9/4 - 5/4
    collab_tot=df_code['collaboration']= (0*df_code['QC1'].fillna(value=3)+ 0*df_code['QC2'].fillna(value=3)+.59*df_code['QC3'].fillna(value=3)+.41*df_code['QC4'].fillna(value=3))* 9/4 - 5/4
    perf_tot=df_code['perfection']= (.55*(6-df_code['QP3'].fillna(value=3))+.45*df_code['QP4'].fillna(value=3))* 9/4 - 5/4
    czx_tot=df_code['CZX']= (df_code['CZ'].fillna(value=2.5)-1)*3+1
    iz_tot=df_code['IZ']= (.47*df_code['CZX']/2 + .53*df_code['SDR'].fillna(value=3)) * 9/4- 5/4; 
    score_tot=df_code['score']=(df_code['trust']+df_code['res']+df_code['div']+df_code['belief']+df_code['collaboration']+df_code['perfection']+df_code['IZ'])/7






def main():

    if individual != '0':
        workgroup=False
        if ind_in_workgroup != '0':
            wg_code=df0.iloc[0]['Code']
            if math.isnan(wg_code):
                print "No workgroup Associated w individual"
                return                
            bii_group_data(code)
            bii_radar1(individual,workgroup)
            bii_radar2(individual,workgroup)
            bii_vbar1(i,wg)
            bii_hist(i,wg)
            



wg=False
i='andrew_k_yu@yahoo.com'
i='yadavsuren@gmail.com'
i='zhirshfeld@gmail.com'
i='jegoubet@gmail.com'

i=False
wg = 'Total'
wg='ELPP-2015F'
wg='BM-BC-2016S'



# Create data and plot
bii_data(i,wg)
bii_radar1(i,wg)
bii_radar2(i,wg)
bii_vbar1(i,wg)
bii_hist(i,wg)



#bii_hbar1(i,wg)





### READ IN DATA and Specify data variables


















import matplotlib.pyplot as plt
import pandas as pd
from pylab import *
import os
print os.getcwd()
import numpy as np
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2

    def draw_poly_patch(self):
        verts = unit_poly_verts(theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts


##Is perfection right czx and collab?!!? + str(df0['Email Address'][i] + str(df0['Email Address'][j]



# Take out Total score and write it underneath the graphs

# PLOT RADAR 1

def bii_radar1(i,w):
    
    def create_data(i,w):
        if i != False:
            data = [
                    ['Trust', 'Res', 'Div', 'MS','Perf', 'Collab', 'RA', 'IZ'],
                    (i, [
                        [float(trust),float(res),float(div),float(bel),float(collab),float(perf),float(czx),float(iz)],
                        [mean(trust_tot),mean(res_tot),mean(div_tot),mean(bel_tot),mean(collab_tot),mean(perf_tot),mean(czx_tot),mean(iz_tot)]])
                    ]
            return data
        elif wg != False:
            data = [
                    ['Trust', 'Res', 'Div', 'MS','Perf', 'Collab', 'RA', 'IZ'],
                    (wg, [
                        [mean(trust_tot),mean(res_tot),mean(div_tot),mean(bel_tot),mean(collab_tot),mean(perf_tot),mean(czx_tot),mean(iz_tot)]])
                    ]
            return data

    data=create_data(i,wg)
    N = 8
    theta = radar_factory(N, frame='polygon')
    
    data = data
    spoke_labels = data.pop(0)
    
    fig = plt.figure(figsize=(9, 9))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    
    colors = ['b', 'r', 'g', 'm', 'y']
    # Plot the four cases from the example data on separate axes
    for n, (title, case_data) in enumerate(data):
        ax = fig.add_subplot(2, 2, n + 1, projection='radar')
        plt.rgrids([2, 4, 6, 8, 10])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25)
        ax.set_varlabels(spoke_labels)
        ax.set_ylim([0,10])
    
    # add legend relative to top-left plot
    if i != False:
        plt.subplot(2, 2, 1)
        labels = ('Individual', 'Average total data')
        legend = plt.legend(labels, loc=(0.95, .95), labelspacing=0.1)
        plt.setp(legend.get_texts(), fontsize='small')
    else:
        plt.subplot(2, 2, 1)
        labels = ('Average','WG Area')
        legend = plt.legend(labels, loc=(0.95, .95), labelspacing=0.1)
        plt.setp(legend.get_texts(), fontsize='small')
            
    
    plt.figtext(0.5, 0.965, 'Innovation Index Results',
                ha='center', color='black', weight='bold', size='large')
    plt.xlabel(r'$\mathrm{Innovation \ Index \ Score:}\ %.3f$' %(mean(score)),fontsize='18', fontweight='bold')
    plt.show()




#######

# RADAR EXAMPLE 2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection

def _radar_factory(num_vars):
    theta = 2*np.pi * np.linspace(0, 1-1./num_vars, num_vars)
    theta += np.pi/2

    def unit_poly_verts(theta):
        x0, y0, r = [0.5] * 3
        verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
        return verts

    class RadarAxes(PolarAxes):
        name = 'radar'
        RESOLUTION = 1

        def fill(self, *args, **kwargs):
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(theta * 180/np.pi, labels)

        def _gen_axes_patch(self):
            verts = unit_poly_verts(theta)
            return plt.Polygon(verts, closed=True, edgecolor='k')

        def _gen_axes_spines(self):
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            verts.append(verts[0])
            path = Path(verts)
            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta

def radar_graph(i_string,labels = [], values = [], optimum = []):
    N = len(labels) 
    theta = _radar_factory(N)
    max_val = max(10, 10)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='radar')
    ax.plot(theta, values, color='blue')
    ax.fill(theta, values, facecolor='blue', alpha=0.25)
    ax.plot(theta, optimum, color='r')
    ax.set_varlabels(labels)
    ax.set_ylim([0,10])
    #legend
    import matplotlib.patches as mpatches
    blue_patch = mpatches.Patch(color='blue', label='Individual Score')
    red_patch = mpatches.Patch(color='red', label='Total Average Score ')
    plt.legend(handles=[blue_patch,red_patch],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.figtext(0.55, 1.05, 'Results for '+ i_string,
    ha='center', color='black', size='large',fontweight='bold')
    plt.xlabel(r'$\mathrm{Innovation \ Index \ Score:}\ %.3f$' %(mean(score)),fontsize='18', fontweight='bold')
    plt.show()


### PLOT RADAR

def bii_radar2(i,wg):
    if i != False:
        i_string = i
    else:
        i_string = wg
    
    labels = ['Trust', 'Resilience', 'Diver', 'Mentality','Perfection', 'Collaboration', 'Resource All', 'IZ']
    values = [mean(trust),mean(res),mean(div),mean(bel),mean(collab),mean(perf),mean(czx),mean(iz)]
    optimum = [mean(trust_tot),mean(res_tot),mean(div_tot),mean(bel_tot),mean(collab_tot),mean(perf_tot),mean(czx_tot),mean(iz_tot)]        
    radar_graph(i_string,labels, values, optimum)









## Histogram plotting



from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# read data from a text file. One number per line

def bii_hist(i,wg):
    if i != False:
        i = False
    else:           

        datos=score
        
        # best fit of data
        (mu, sigma) = norm.fit(datos)
        
        # the histogram of the data
        n, bins, patches = plt.hist(datos, 10, normed=1, facecolor='green', alpha=0.75)
        
        # add a 'best fit' line
        y = mlab.normpdf( bins, mu, sigma)
        l = plt.plot(bins, y, 'r--', linewidth=2)
        
        #plot
        plt.xlabel('Innovation index score for' + wg)
        plt.ylabel('Probability')
        plt.title(r'$\mathrm{Overall \ Innovation \ Index \ Score:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
        plt.grid(True)
        plt.plot((mu, mu), (0, 0.45), 'b--')
        
        
        plt.show()












## Bar chart example ADD STANARD DEVIATION TO LEGEND

import numpy as np
import matplotlib.pyplot as plt

def bii_vbar1(i,wg):
    if i != False:
        i_string = i
    else:
        i_string = wg
    
        N = 8
        values = [mean(trust),mean(res),mean(div),mean(bel),mean(collab),mean(perf),mean(czx),mean(iz)]
        valStd = [std(trust),std(res),std(div),std(bel),std(collab),std(perf),std(czx),std(iz)]
        
        
        ind = np.arange(N)  # the x locations for the groups
        width = 0.35       # the width of the bars
        
        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, values, width, color='b', yerr=valStd)
        
        optimum = [mean(trust_tot),mean(res_tot),mean(div_tot),mean(bel_tot),mean(collab_tot),mean(perf_tot),mean(czx_tot),mean(iz_tot)]
        optimStd = [std(trust_tot),std(res_tot),std(div_tot),std(bel_tot),std(collab_tot),std(perf_tot),std(czx_tot),std(iz_tot)]
        
        rects2 = ax.bar(ind + width, optimum, width, color='y', yerr=optimStd)
        
        # add some text for labels, title and axes ticks
        ax.set_ylabel('Score')
        #ax.set_title('Results for ' + i_string,loc='left')
        ax.set_xticks(ind + width)
        ax.set_xticklabels(('Trust', 'Res', 'Div', 'MS','Perf', 'Collab', 'RA', 'IZ'))
        ax.set_ylim([0,10])
        
        
        ax.legend((rects1[0], rects2[0]), ('Individual', 'Mean'),bbox_to_anchor=(1.3, 1.3),borderaxespad=0.)
        
        
        def autolabel(rects):
            # attach some text labels
            for rect in rects:
                height = float(rect.get_height())
                ax.text(float(rect.get_x()) + rect.get_width()/2.0, 1.02*float(height),
                        round(height,1),
                        ha='center', va='bottom',fontsize=9)
        
        autolabel(rects1)
        autolabel(rects2)
        plt.figtext(0.52, 1.02, 'Results for '+ i_string,
        ha='center', color='black', size='large',fontweight='bold')
        plt.xlabel(r'$\mathrm{Innovation \ Index \ Score:}\ %.3f$' %(mean(score)),fontsize='18', fontweight='bold')
        plt.show()










### Lying barchart 1



def bii_hbar1(i,wg):
    if i != False:
        i=False
    else:

        if i != False:
            i_string = i
        else:
            i_string = wg
            
        import matplotlib.pyplot as plt
        
        val = [mean(trust),mean(res),mean(div),mean(bel),mean(collab),mean(perf),mean(czx),mean(iz)]
        err = [std(trust),std(res),std(div),std(bel),std(collab),std(perf),std(czx),std(iz)]
        pos = arange(8)    # the bar centers on the y axis


        
        plt.barh(pos,val, xerr=err, ecolor='r', align='center')
        plt.yticks(pos, (('Trust', 'Res', 'Div', 'MS','Perf', 'Collab', 'RA', 'IZ')))
        plt.xlabel('Score')
        plt.plot((mean(score), mean(score)), (-1, 8), 'g')
        plt.title('Score for ' + i_string)
        plt.xlabel(r'$\mathrm{Innovation \ Index \ Score:}\ %.3f$' %(mean(score)),fontsize='18', fontweight='bold')
        axes = plt.gca()
        axes.set_xlim([0,10])
        plt.show()
    
    
#
#def bii_hbar2(i,wg):
#
#
#import numpy as np
#import matplotlib.pyplot as plt
#import pylab
#from matplotlib.ticker import MaxNLocator
#
#if i != False:
#    i_string = i
#else:
#    i_string = wg
#
#student = 'Results for ' + i_string
#grade = 2
#gender = 'boy'
#cohortSize = 62  # The number of other 2nd grade boys
#
#numTests = 9
#testNames = ['Trust', 'Res', 'Div', 'MS','Perf', 'Collab', 'RA', 'IZ']
#testMeta = ['', '', '', '', '']
#scores = [mean(trust),mean(res),mean(div),mean(bel),mean(collab),mean(perf),mean(czx),mean(iz)]
#rankings = np.round(np.random.uniform(0, 1, numTests)*100, 0)
#
#
#fig, ax1 = plt.subplots(figsize=(9, 7))
#plt.subplots_adjust(left=0.115, right=0.88)
#fig.canvas.set_window_title('Eldorado K-8 Fitness Chart')
#pos = np.arange(numTests)+0.5    # Center bars on the Y-axis ticks
#rects = ax1.barh(pos, rankings, align='center', height=0.5, color='m')
#
#ax1.axis([0, 100, 0, 5])
#pylab.yticks(pos, testNames)
#ax1.set_title('Johnny Doe')
#plt.text(50, -0.5, 'Cohort Size: ' + str(cohortSize),
#         horizontalalignment='center', size='small')
#
## Set the right-hand Y-axis ticks and labels and set X-axis tick marks at the
## deciles
#ax2 = ax1.twinx()
#ax2.plot([100, 100], [0, 5], 'white', alpha=0.1)
#ax2.xaxis.set_major_locator(MaxNLocator(11))
#xticks = pylab.setp(ax2, xticklabels=['0', '10', '20', '30', '40', '50', '60',
#                                      '70', '80', '90', '100'])
#ax2.xaxis.grid(True, linestyle='--', which='major', color='grey',
#alpha=0.25)
##Plot a solid vertical gridline to highlight the median position
#plt.plot([50, 50], [0, 5], 'grey', alpha=0.25)
#
## Build up the score labels for the right Y-axis by first appending a carriage
## return to each string and then tacking on the appropriate meta information
## (i.e., 'laps' vs 'seconds'). We want the labels centered on the ticks, so if
## there is no meta info (like for pushups) then don't add the carriage return to
## the string
#
#
#def withnew(i, scr):
#    if testMeta[i] != '':
#        return '%s\n' % scr
#    else:
#        return scr
#
#scoreLabels = [withnew(i, scr) for i, scr in enumerate(scores)]
#scoreLabels = [i+j for i, j in zip(scoreLabels, testMeta)]
## set the tick locations
#ax2.set_yticks(pos)
## set the tick labels
#ax2.set_yticklabels(scoreLabels)
## make sure that the limits are set equally on both yaxis so the ticks line up
#ax2.set_ylim(ax1.get_ylim())
#
#
#ax2.set_ylabel('Test Scores')
##Make list of numerical suffixes corresponding to position in a list
##            0     1     2     3     4     5     6     7     8     9
#suffixes = ['th', 'st', 'nd', 'rd', 'th', 'th', 'th', 'th', 'th', 'th']
#ax2.set_xlabel('Percentile Ranking Across ' + str(grade) + suffixes[grade]
#              + ' Grade ' + gender.title() + 's')
#
## Lastly, write in the ranking inside each bar to aid in interpretation
#for rect in rects:
#    # Rectangle widths are already integer-valued but are floating
#    # type, so it helps to remove the trailing decimal point and 0 by
#    # converting width to int type
#    width = int(rect.get_width())
#
#    # Figure out what the last digit (width modulo 10) so we can add
#    # the appropriate numerical suffix (e.g., 1st, 2nd, 3rd, etc)
#    lastDigit = width % 10
#    # Note that 11, 12, and 13 are special cases
#    if (width == 11) or (width == 12) or (width == 13):
#        suffix = 'th'
#    else:
#        suffix = suffixes[lastDigit]
#
#    rankStr = str(width) + suffix
#    if (width < 5):        # The bars aren't wide enough to print the ranking inside
#        xloc = width + 1   # Shift the text to the right side of the right edge
#        clr = 'black'      # Black against white background
#        align = 'left'
#    else:
#        xloc = 0.98*width  # Shift the text to the left side of the right edge
#        clr = 'white'      # White on magenta
#        align = 'right'
#
#    # Center the text vertically in the bar
#    yloc = rect.get_y()+rect.get_height()/2.0
#    ax1.text(xloc, yloc, rankStr, horizontalalignment=align,
#            verticalalignment='center', color=clr, weight='bold')
#
#plt.show()
#    
#    
#    



if (__name__ == "__main__"):
    main()
    
    
    
    
    
# Start by reading in the full data
#
#df0 = pd.read_csv('fulldata.csv')
#trust_tot=df0['trust']=(.27*df0['QT1'].fillna(value=3)+.16*(6-df0['QT2'].fillna(value=3))+.34*df0['QT3'].fillna(value=3)+.22*(6-df0['QT1'].fillna(value=3)))* 9/4 - 5/4
#res_tot=df0['res']=(.26*df0['QF1'].fillna(value=3)+.21*df0['QF2'].fillna(value=3)+.35*df0['QF3'].fillna(value=3)+.17*df0['QF4'].fillna(value=3))* 9/4 - 5/4
#div_tot=df0['div']= (.15*df0['QD1'].fillna(value=3)+.29*df0['QD2'].fillna(value=3)+.35*df0['QD3'].fillna(value=3)+.20*df0['QD4'].fillna(value=3))* 9/4 - 5/4
#bel_tot=df0['belief']=(.24*df0['QB1'].fillna(value=3)+.3*df0['QB2'].fillna(value=3)+.3*df0['QB3'].fillna(value=3)+.16*df0['QB4'].fillna(value=3))* 9/4 - 5/4
#collab_tot=df0['collaboration']= (0*df0['QC1'].fillna(value=3)+ 0*df0['QC2'].fillna(value=3)+.59*df0['QC3'].fillna(value=3)+.41*df0['QC4'].fillna(value=3))* 9/4 - 5/4
#perf_tot=df0['perfection']= (.55*(6-df0['QP3'].fillna(value=3))+.45*df0['QP4'].fillna(value=3))* 9/4 - 5/4
#czx_tot=df0['CZX']= (df0['CZ'].fillna(value=2.5)-1)*3+1
#iz_tot=df0['IZ']= (.47*df0['CZX']/2 + .53*df0['SDR'].fillna(value=3)) * 9/4- 5/4; 
#score_tot=df0['score']=(df0['trust']+df0['res']+df0['div']+df0['belief']+df0['collaboration']+df0['perfection']+df0['IZ'])/7

