from contextlib import contextmanager
from copy import deepcopy
from functools import reduce
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import PIL
import plotnine
from plotnine import theme

import potoo.mpl_backend_xee
from potoo.util import puts


def plot_set_defaults():
    figsize()
    plot_set_default_mpl_rcParams()


# ~/.matploblib/matploblibrc isn't read by ipykernel, so we call this from ~/.pythonrc which is
#   - TODO What's the right way to init mpl.rcParams?
def plot_set_default_mpl_rcParams():
    mpl.rcParams['figure.facecolor'] = 'white'  # Match savefig.facecolor
    mpl.rcParams['image.interpolation'] = 'nearest'  # Don't interpolate, show square pixels
    # TODO Sync more carefully with ~/.matploblib/matploblibrc?


def theme_figsize(width=8, aspect_ratio=2/3, dpi=72):
    """
    plotnine theme with defaults for figure_size width + aspect_ratio (which overrides figure_size height if defined):
    - https://plotnine.readthedocs.io/en/stable/generated/plotnine.themes.theme.html#aspect_ratio-and-figure_size
    """
    return theme(
        # height is ignored by plotnine when aspect_ratio is given, but supply anyway so that we can set theme.rcParams
        # into mpl.rcParams since the latter has no notion of aspect_ratio [TODO Submit PR to fix]
        figure_size=[width, width * aspect_ratio],
        aspect_ratio=aspect_ratio,
        dpi=dpi,
    )


def figsize(*args, **kwargs):
    """
    Set theme_figsize(...) as global plotnine.options + mpl.rcParams:
    - https://plotnine.readthedocs.io/en/stable/generated/plotnine.themes.theme.html
    - http://matplotlib.org/users/customizing.html
    """
    t = theme_figsize(*args, **kwargs)
    mpl.rcParams.update(t.rcParams)
    plotnine.options.figure_size = t.themeables['figure_size'].properties['value']
    plotnine.options.aspect_ratio = t.themeables['aspect_ratio'].properties['value']
    plotnine.options.dpi = t.themeables['dpi'].properties['value']  # (Ignored for figure_format='svg')


#
# mpl
#


def plot_plt(
    passthru=None,
    tight_layout=plt.tight_layout,
):
    tight_layout and tight_layout()
    plt.show()
    return passthru


def plot_img(data, basename_suffix=''):
    data = _plot_img_cast(data)
    with potoo.mpl_backend_xee.basename_suffix(basename_suffix):
        return potoo.mpl_backend_xee.imsave_xee(data)


def _plot_img_cast(x):
    try:
        import IPython.core.display
        ipython_image_type = IPython.core.display.Image
    except:
        ipython_image_type = ()  # Empty tuple of types for isinstance, always returns False
    if isinstance(x, ipython_image_type):
        return PIL.Image.open(x.filename)
    else:
        return x


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


#
# ggplot
#   - Docs are incomplete but have some helpful examples: http://yhat.github.io/ggplot/docs.html
#   - Use the source for actual reference: https://github.com/yhat/ggplot
#


def plot_gg(
    g,
    tight_layout=None,  # Already does some sort of layout tightening, doing plt.tight_layout() makes it bad
):
    # TODO g.title isn't working? And is clobbering basename_suffix somehow?
    # with potoo.mpl_backend_xee.basename_suffix(g.title or potoo.mpl_backend_xee.basename_suffix.value()):
    # repr(g)  # (Over)optimized for repl/notebook usage (repr(g) = g.make(); plt.show())
    # TODO HACK Why doesn't theme_base.__radd__ allow multiple themes to compose?
    if not isinstance(g.theme, theme_rc):
        g += theme_rc({}, g.theme)
    g.make()
    tight_layout and tight_layout()
    plt.show()
    # return g # Don't return to avoid plotting a second time if repl/notebook


def gg_sum(*args):
    "Simpler syntax for lots of ggplot '+' layers when you need to break over many lines, comment stuff out, etc."
    return reduce(lambda a, b: a + b, args)


# TODO Update for plotnine
# import ggplot as gg
# class theme_rc(gg.themes.theme):
#     '''
#     - Avoids overriding key defaults from ~/.matplotlib/matplotlibrc (including figure.figsize)
#     - Allows removing existing mappings by adding {k: None}
#     - Avoids mutating the global class var theme_base._rcParams
#     - Hacks up a way to compose themes (base themes don't compose, and they also dirty shared global state)
#     - Can also passthru kwargs to gg.themes.theme, e.g. x_axis_text=attrs(kwargs=dict(rotation=45))
#     '''
#
#     def __init__(
#         self,
#         rcParams={},
#         theme=gg.theme_gray(),  # HACK: Copy default from ggplot.theme
#         **kwargs
#     ):
#         super(theme_rc, self).__init__(**kwargs)
#         rcParams.setdefault('figure.figsize', None)  # Use default, e.g. from ~/.matplotlib/matplotlibrc
#         self.rcParams = rcParams  # Don't mutate global mutable class var theme_base._rcParams
#         self.theme = theme
#
#     def __radd__(self, other):
#         self.theme.__radd__(other)                    # Whatever weird side effects
#         return super(theme_rc, self).__radd__(other)  # Our own weird side effects
#
#     def get_rcParams(self):
#         return {
#             k: v
#             for k, v in dict(self.theme.get_rcParams(), **self.rcParams).items()
#             if v is not None  # Remove existing mapping
#         }
#
#     def apply_final_touches(self, ax):
#         return self.theme.apply_final_touches(ax)
#
#
# import ggplot as gg
# class scale_color_cmap(gg.scales.scale.scale):
#     '''
#     ggplot scale from a mpl colormap, e.g.
#
#         scale_color_cmap('Set1')
#         scale_color_cmap(plt.cm.Set1)
#
#     Docs: http://matplotlib.org/users/colormaps.html
#     '''
#
#     def __init__(self, cmap):
#         self.cmap = cmap if isinstance(cmap, mpl.colors.Colormap) else plt.cm.get_cmap(cmap)
#
#     def __radd__(self, gg):
#         color_col = gg._aes.data.get('color', gg._aes.data.get('fill'))
#         n_colors = 3 if not color_col else max(gg.data[color_col].nunique(), 3)
#         colors = [self.cmap(x) for x in np.linspace(0, 1, n_colors)]
#         gg.colormap = self.cmap        # For aes(color=...) + continuous
#         gg.manual_color_list = colors  # For aes(color=...) + discrete
#         gg.manual_fill_list = colors   # For aes(fill=...)  + discrete
#         # ...                          # Any cases I've missed?
#         return gg


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


#
# seaborn
#


def plot_sns(passthru=None):
    # TODO No way to set figsize for all seaborn plots? e.g. sns.factorplot(size, aspect) always changes figsize
    plt.show()
    return passthru


def sns_size_aspect(
    rows=1,
    cols=1,
    scale=1,
    figsize=np.array(mpl.rcParams['figure.figsize']),
):
    '''
    e.g. http://seaborn.pydata.org/generated/seaborn.factorplot.html
    '''
    (figw, figh) = figsize
    rowh = figh / rows * scale
    colw = figw / cols * scale
    return dict(
        size   = rowh,
        aspect = colw / rowh,
    )
