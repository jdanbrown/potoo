from contextlib import contextmanager
from copy import deepcopy
from functools import reduce
import ggplot as gg
import matplotlib as mpl
import matplotlib.pyplot as plt

import jdanbrown.mpl_backend_xee
from jdanbrown.util import puts


mpl.style.use('ggplot')


class gg_xtight(object):

    def __init__(self, margin=0.05):
        self.margin = margin

    def __radd__(self, g):
        g          = deepcopy(g)
        xs         = g.data[g._aes['x']]
        lims       = [xs.min(), xs.max()]
        margin_abs = float(self.margin) * (lims[1] - lims[0])
        g.xlimits  = [xs.min() - margin_abs, xs.max() + margin_abs]
        return g


class gg_ytight(object):

    def __init__(self, margin=0.05):
        self.margin = margin

    def __radd__(self, g):
        g          = deepcopy(g)
        ys         = g.data[g._aes['y']]
        lims       = [ys.min(), ys.max()]
        margin_abs = float(self.margin) * (lims[1] - lims[0])
        g.ylimits  = [ys.min() - margin_abs, ys.max() + margin_abs]
        return g


class gg_tight(object):

    def __init__(self, margin=0.05):
        self.margin = margin

    def __radd__(self, g):
        return g + gg_xtight(self.margin) + gg_ytight(self.margin)


def gg_layer(*args):
    'More uniform syntax than \ and + for many-line layer addition'
    return reduce(lambda a, b: a + b, args)


class gg_theme_keep_defaults_for(gg.theme_gray):
    def __init__(self, *rcParams):
        super(gg_theme_keep_defaults_for, self).__init__()
        for x in rcParams:
            del self._rcParams[x]


def plot_gg(ggplot):
    ggplot += gg_theme_keep_defaults_for('figure.figsize')
    # TODO ggplot.title isn't working? And is clobbering basename_suffix somehow?
    # with jdanbrown.mpl_backend_xee.basename_suffix(ggplot.title or jdanbrown.mpl_backend_xee.basename_suffix.value()):
    repr(ggplot)  # (Over)optimized for repl/notebook usage (repr(ggplot) = ggplot.make(); plt.show())
    # return ggplot # Don't return to avoid plotting a second time if repl/notebook


def plot_sns(g):
    # TODO Duplicated from ~/.matplotlib/matplotlibrc; better way to do this?
    g.fig.set_figheight(12.24)
    g.fig.set_figwidth(10.5)
    g.fig.set_dpi(80)
    plt.show()
    return g


def plot_plt(
    passthru=None,
    tight_layout=plt.tight_layout,
):
    # TODO improt seaborn messes up mpl/plt defaults :(
    # g.fig.set_figheight(12.24)
    # g.fig.set_figwidth(10.5)
    # g.fig.set_dpi(80)
    tight_layout and tight_layout()
    plt.show()
    return passthru


def plot_img(data, basename_suffix=''):
    with jdanbrown.mpl_backend_xee.basename_suffix(basename_suffix):
        return jdanbrown.mpl_backend_xee.imsave_xee(data)


def plot_img_via_imshow(data):
    'Makes lots of distorted pixels, huge PITA, use imsave/plot_img instead'
    (h, w) = data.shape[:2]  # (h,w) | (h,w,3)
    dpi    = 100
    k      = 1  # Have to scale this up to ~4 to avoid distorted pixels
    with tmp_rcParams({
        'image.interpolation': 'nearest',
        'figure.figsize':      puts((w/float(dpi)*k, h/float(dpi)*k)),
        'savefig.dpi':         dpi,
    }):
        img = plt.imshow(data)
        img.axes.get_xaxis().set_visible(False)  # Don't create padding for axes
        img.axes.get_yaxis().set_visible(False)  # Don't create padding for axes
        plt.axis('off')                          # Don't draw axes
        plt.tight_layout(pad=0)                  # Don't add padding
        plt.show()


@contextmanager
def tmp_rcParams(kw):
    _save_mpl = mpl.RcParams(**mpl.rcParams)
    _save_plt = mpl.RcParams(**plt.rcParams)
    try:
        mpl.rcParams.update(kw)
        plt.rcParams.update(kw)  # TODO WHY ARE THESE DIFFERENT
        yield
    finally:
        mpl.rcParams = _save_mpl
        plt.rcParams = _save_plt
