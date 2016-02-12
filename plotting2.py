# -*- coding: utf-8 -*-
"""
Created on Feb 1st 2016

@author: alexander.fred-ojala
"""

import matplotlib.pyplot as plt
import pandas as pd
from pylab import *
import os
print os.getcwd()


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

#Calculate overall score
score=df0['score']=(df0['trust']+df0['res']+df0['div']+df0['belief']+df0['collaboration']+df0['perfection']+df0['IZ'])/7

#Check labels in data fram
print df0[df0.PROJECT=='ARN16S']



### PLOTTING EXAMPLES


## Radar chart


import numpy as np

import matplotlib.pyplot as plt
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
def create_data(i,j):
    data = [
            ['Trust', 'Res', 'Div', 'MS','Perf', 'Collab', 'RA', 'IZ','Score'],
            (str(df0['Email Address'][i]), [
                [float(trust[i]),float(res[i]),float(div[i]),float(bel[i]),float(collab[i]),float(perf[i]),float(czx[i]),float(iz[i]), float(score[i])],
                [mean(trust),mean(res),mean(div),mean(bel),mean(collab),mean(perf),mean(czx),mean(iz), mean(score)]]),
            (str(df0['Email Address'][j]) , [
                [trust[j],res[j],div[j],bel[j],collab[j],perf[j],czx[j],iz[j], score[j]],
                [mean(trust),mean(res),mean(div),mean(bel),mean(collab),mean(perf),mean(czx),mean(iz), mean(score)]])
            ]
    return data


# Take out Total score and write it underneath the graphs

# PLOT RADAR 1
data=create_data(34,29)
N = 9
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
plt.subplot(2, 2, 1)
labels = ('Individual', 'Mean')
legend = plt.legend(labels, loc=(0.95, .95), labelspacing=0.1)
plt.setp(legend.get_texts(), fontsize='small')

plt.figtext(0.5, 0.965, 'Innovation Index Results',
            ha='center', color='black', weight='bold', size='large')
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

def radar_graph(i,labels = [], values = [], optimum = []):
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
    red_patch = mpatches.Patch(color='red', label='Overall mean')
    plt.legend(handles=[blue_patch,red_patch],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.figtext(0.5, 0.965, 'Innovation Index Results for '+ str(df0['Email Address'][i]),
    ha='center', color='black', weight='bold', size='large')
    plt.show()
    plt.savefig("radar.png", dpi=100)


### PLOT RADAR

i=23
labels = ['Trust', 'Resilience', 'Diver', 'Mentality','Perfection', 'Collaboration', 'Resource All', 'IZ','Total Score']
values = [float(trust[i]),float(res[i]),float(div[i]),float(bel[i]),float(collab[i]),float(perf[i]),float(czx[i]),float(iz[i]), float(score[i])]
optimum = [mean(trust),mean(res),mean(div),mean(bel),mean(collab),mean(perf),mean(czx),mean(iz), mean(score)]

radar_graph(i,labels, values, optimum)









## Histogram plotting



from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# read data from a text file. One number per line
datos=score

# best fit of data
(mu, sigma) = norm.fit(datos)

# the histogram of the data
n, bins, patches = plt.hist(datos, 60, normed=1, facecolor='green', alpha=0.75)

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)

#plot
plt.xlabel('Indiviuals innovation index score')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Overall \ Innovation \ Index \ Score:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
plt.grid(True)
plt.plot((mu, mu), (0, 0.6), 'b--')

plt.show()












## Bar chart example

import numpy as np
import matplotlib.pyplot as plt

N = 9
values = [float(trust[i]),float(res[i]),float(div[i]),float(bel[i]),float(collab[i]),float(perf[i]),float(czx[i]),float(iz[i]), float(score[i])]
valStd = (0,0,0,0,0,0,0,0,0)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, values, width, color='b', yerr=valStd)

optimum = [mean(trust),mean(res),mean(div),mean(bel),mean(collab),mean(perf),mean(czx),mean(iz), mean(score)]
optimStd = (1,2,1.5,3.2,2,1,1,0,1.2)
rects2 = ax.bar(ind + width, optimum, width, color='y', yerr=optimStd)

# add some text for labels, title and axes ticks
ax.set_ylabel('Score')
ax.set_title('Innovation Index Score')
ax.set_xticks(ind + width)
ax.set_xticklabels(('Trust', 'Res', 'Div', 'MS','Perf', 'Collab', 'RA', 'IZ','Score'))
ax.set_ylim([0,10])


ax.legend((rects1[0], rects2[0]), ('Individual', 'Mean'),bbox_to_anchor=(1.15, 1.1),borderaxespad=0.)


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = float(rect.get_height())
        ax.text(float(rect.get_x()) + rect.get_width()/2.0, 1.05*float(height),
                round(height,1),
                ha='center', va='bottom',fontsize=9)
        print round(height,1)

autolabel(rects1)
autolabel(rects2)

plt.show()