# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 15:06:04 2016

@author: FO
"""

# Radar 1

import matplotlib.pyplot as plt
from pylab import *
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
    
    def create_data(ident,code):
        if ident == 1:
            data = [
                    ['Trust', 'Res', 'Div', 'Ment Str','Res All', 'Collab', 'Com Zone', 'In Zone'],
                    (code, [
                        [float(trust),float(res),float(div),float(bel),float(collab),float(resall),float(czx),float(iz)],
                        [mean(trust_tot),mean(res_tot),mean(div_tot),mean(bel_tot),mean(collab_tot),mean(resall_tot),mean(czx_tot),mean(iz_tot)]])
                    ]
            return data
        elif wg != False:
            data = [
                    ['Trust', 'Res', 'Div', 'Ment Str','Res All', 'Collab', 'Com Zone', 'In Zone'],
                    (code, [
                        [mean(trust_tot),mean(res_tot),mean(div_tot),mean(bel_tot),mean(collab_tot),mean(resall_tot),mean(czx_tot),mean(iz_tot)]])
                    ]
            return data

    data=create_data(ident,code)
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