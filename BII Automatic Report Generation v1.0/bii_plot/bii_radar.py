import matplotlib
matplotlib.use('Agg')
from sys import argv
import numpy as np
import pandas as pd
from pylab import *
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
import matplotlib.patches as mpatches



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

def graph_data(in_data, in_data_comp, code, comp):
    # Here we would like to include if it is an individual or a workgroup? Maybe not. Because the output will be the same
    # If we would like some comparison data this is easily done by adding more data and an if-statement
    [trust,res,div,bel,collab,resall,czx,comfort,iz,score] = in_data
    if in_data_comp is None:
        data = [
            ['Trust', 'Res', 'Div', 'Ment Str', 'Collab', 'Res All', 'Com Zone', 'In Zone'],
            (code, [
                [mean(trust), mean(res), mean(div), mean(bel), mean(collab), mean(resall), mean(comfort), mean(iz)]])
        ]
    else:
        [t, r, d, mb, col, ra, czxc, cz, iz, score] = in_data_comp
        data = [
        ['Trust', 'Res', 'Div', 'Ment Str', 'Collab', 'Res All', 'Com Zone', 'In Zone'],
        (code, [
            [mean(trust), mean(res), mean(div), mean(bel), mean(collab), mean(resall), mean(comfort), mean(iz)],
            [mean(t), mean(r), mean(d), mean(mb), mean(col), mean(ra), mean(cz), mean(iz)]])
        ]
    return data

def bii_radar(ident, code, data_bii, data_comp, comp, wg, proj):
    score = mean(data_bii[9])

    if len(code) == 2 and not isinstance(code, basestring):
        code = code[0] + " " + code[1]

    data = graph_data(data_bii, data_comp, code, comp)
    
    N = 8
    theta = radar_factory(N, frame='polygon')

    spoke_labels = data.pop(0)
    # spoke labels removes the first entry in the data, i.e. the labels

    fig = plt.figure(figsize=(5.2,5.2))
    #fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    colors = ['b', 'r', 'g', 'm', 'y']



    for n, (title, case_data) in enumerate(data):
        ax = fig.add_subplot(111, projection='radar')
        plt.rgrids([2, 4, 6, 8, 10])
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25)
        ax.set_varlabels(spoke_labels)
        ax.set_ylim([0,10])

    # add legend relative to top-left plot
    plt.subplot(111)
    
    #label = ('Score')
    #legend = plt.legend(label, loc=(0.82, .95), labelspacing=0.1)
    #plt.setp(legend.get_texts(), fontsize='small')

    lbl_code=code

    if wg is None and proj is None:
        lbl_code = 'Score'

    blue_patch = mpatches.Patch(color='b', alpha=0.25, label=lbl_code)

    if data_comp is None:
        lgd = plt.legend(handles=[blue_patch], loc=(0.78  , .95))
        #When comparison is added just do another patch and add it to the legend.
    else:
        red_patch = mpatches.Patch(color='r', alpha=0.25, label=comp)
        lgd = plt.legend(handles=[blue_patch, red_patch], loc=(0.78  , .95))

    title_name = 'Results for %s' %code


    tit = plt.suptitle(title_name, fontsize=14, fontweight='bold', y=1.05)
        #plt.figtext(0.5, 0.965, title_name,
         #   ha='center', color='black', weight='bold', size='large')
    plt.xlabel(r'$\mathrm{Innovation \ Index \ Score:}\ %.3f$' %(mean(score)),fontsize='18', fontweight='bold')


    file_name = "radar"
    path_name = "static/%s" %file_name

    #path_name = "/Users/johanenglarsson/bii/mod/comp/static/%s" %file_name
    plt.tight_layout()
    #plt.savefig(path_name)
    plt.savefig(path_name, bbox_extra_artists=(lgd,tit,))
    





