import gc
import os
import random
import signal
import time

from attrdict import AttrDict
from dataclasses import dataclass
from IPython.core.getipython import get_ipython
from IPython.display import *
import numpy as np
import pandas as pd
import prompt_toolkit

import potoo.pandas
from potoo.pandas import cat_to_str
from potoo.util import or_else, deep_round_sig, singleton


def ipy_format(*xs: any) -> str:
    """
    Format like IPython.display.display
    - Spec: print(ipy_format(*xs)) ~ display(*xs)
    - Manually line-join multiple args like display(x, y) does
    """
    return '\n'.join(
        formats['text/plain']
        for x in xs
        for formats, metadata in [get_ipython().display_formatter.format(x)]
    )


def is_ipython_console():
    return or_else(None, lambda: get_ipython().__class__.__name__) == 'TerminalInteractiveShell'


def is_ipython_notebook():
    return or_else(None, lambda: get_ipython().__class__.__name__) == 'ZMQInteractiveShell'


def disable_special_control_backslash_handler():
    """
    Replace special ipy binding for C-\ with normal os SIGQUIT handler
    - Since https://github.com/ipython/ipython/pull/9820/commits/37863a8
    """
    ipy = get_ipython()
    if hasattr(ipy, 'pt_cli'):
        ipy.pt_cli.application.key_bindings_registry.add_binding(prompt_toolkit.keys.Keys.ControlBackslash)(
            lambda ev: os.kill(os.getpid(), signal.SIGQUIT)
        )


def set_display_on_ipython_prompt():
    """
    set_display on each ipython prompt: workaround bug where user SIGWINCH handlers are ignored while readline is active
    - https://bugs.python.org/issue23735
    - Doesn't happen in python readline, just ipython readline
    - Happens with python-3.6.0 and python-3.6.2 (latest as of 2017-08-26)
    """
    if is_ipython_console():
        ipy = get_ipython()
        ipy.set_hook(
            'pre_run_code_hook',
            _warn_deprecated=False,
            hook=lambda *args: potoo.pandas.set_display(),
        )


def gc_on_ipy_post_run_cell():
    """
    Force gc after each ipy cell run, since it's _really_ easy to accumulate many uncollected-but-collectable GBs of mem
    pressure by re-executing one heavy cell over and over again
    - Adds ballpark ~40ms per cell execution
    - Use in combination with ipy `c.InteractiveShell.cache_size = 0` (see ~/.ipython/profile_default/ipython_config.py)
    """
    ipy = get_ipython()
    if ipy:  # None if not ipython
        # gc.collect() takes ballpark ~40ms, so avoid running it every single time
        def pre_run_cell(info):
            info.start_s = time.time()
        def post_run_cell(result):
            if hasattr(result.info, 'start_s'):
                elapsed_s = time.time() - result.info.start_s
                if elapsed_s > .5 or random.random() < 1/20:
                    gc.collect()
        ipy.events.register('pre_run_cell', pre_run_cell)
        ipy.events.register('post_run_cell', post_run_cell)


@singleton
@dataclass
class ipy_formats:

    config = AttrDict(
        deep_round_sig=True,  # Very useful by default [and hopefully doesn't break anything...]
        stack_iters=False,  # Useful, but maybe not by default
    )

    @property
    def precision(self):
        return pd.get_option('display.precision')

    # TODO Respect display.max_rows (currently treats it as unlimited)
    def set(self):
        ipy = get_ipython()
        if ipy:

            ipy.display_formatter.formatters['text/html'].for_type(pd.DataFrame, self._format_html_df)

            # TODO 'text/plain' for pd.DataFrame
            #   - TODO Have to self.foo(...) instead of returning... [what did I mean by this a long time ago?]
            # ipy.display_formatter.formatters['text/plain'].for_type(pd.DataFrame, lambda df, self, cycle:
            #     df.apply(axis=0, func=lambda col:
            #         col.apply(lambda x:
            #             x if not isinstance(x, list)
            #             else '' if len(x) == 0
            #             else pd.Series(x).to_string(index=False)
            #         )
            #     ).to_string()
            # )

            # pd.Series doesn't have _repr_html_ or .to_html, like pd.DataFrame does
            #   - TODO 'text/plain' for pd.Series
            #       - I briefly tried and nothing happened, so I went with 'text/html' instead; do more research...
            ipy.display_formatter.formatters['text/html'].for_type(pd.Series, self._format_html_series)

    def _format_html_df(self, df: pd.DataFrame) -> str:
        return df.apply(axis=0, func=lambda col:
            # cat_to_str to avoid .apply mapping all cat values, which we don't need and could be slow for large cats
            col.pipe(cat_to_str).apply(self._format_html_df_cell)
        ).to_html(
            escape=False,  # Allow html in cells
        )

    def _format_html_series(self, s: pd.Series) -> str:
        # Use <div> instad of <pre>, since <pre> might bring along a lot of style baggage
        return '<div style="white-space: pre">%s</div>' % (
            # cat_to_str to avoid .apply mapping all cat values, which we don't need and could be slow for large cats
            s.pipe(cat_to_str).apply(self._format_any),
        )

    def _format_html_df_cell(self, x: any) -> any:

        if self.config.stack_iters and isinstance(x, (list, tuple, np.ndarray)):
            # When displaying df's as html, display list/etc. cells as vertically stacked values (inspired by bq web UI)
            # - http://ipython.readthedocs.io/en/stable/api/generated/IPython.core.formatters.html
            ret = '' if len(x) == 0 else (
                pd.Series(self._format_any(y) for y in x)
                .to_string(index=False)
                .replace('\n', '<br/>')
            )
        else:
            ret = self._format_any(x)

        # HACK A weird thing to help out atom/jupyter styling
        #   - TODO Should this be: _has_number(ret)? _has_number(x)? _has_number(col) from one frame above?
        if not self._has_number(ret):
            ret = '<div class="not-number">%s</div>' % (ret,)

        return ret

    def _format_any(self, x: any) -> any:
        if all([
            self.config.deep_round_sig,
            # Don't touch np.array's since they can be really huge, and numpy is already smart about truncating them
            not isinstance(x, np.ndarray),
        ]):
            return deep_round_sig(x, self.precision)
        else:
            return x

    # XXX Subsumed by deep_round_sig
    # def _format_any(self, x: any) -> any:
    #     if np.issubdtype(type(x), np.complexfloating):
    #         return self._round_to_precision_complex(x)
    #     else:
    #         return x
    #
    # # HACK Pandas by default displays complex values with precision 16, even if you np.set_printoptions(precision=...)
    # # and pd.set_option('display.precision', ...). This is a hack to display like precision=3.
    # #   - TODO Submit bug to pandas
    # #   - TODO Make this more sophisticated, e.g. to reuse (or mimic) the magic in numpy.core.arrayprint
    # #   - Return complex, not str, so that e.g. pd.Series displays 'dtype: complex' and not 'dtype: object'
    # def _round_to_precision_complex(self, z: complex) -> complex:
    #     # Use complex(...) instead of type(z)(...), since complex parses from str but e.g. np.complex128 doesn't.
    #     # Use z.__format___ instead of '%.3g' % ..., since the latter doesn't support complex numbers (dunno why).
    #     return complex(z.__format__('.%sg' % self.precision))

    def _has_number(self, x: any) -> bool:
        return (
            np.issubdtype(type(x), np.number) or
            isinstance(x, list) and any(self._has_number(y) for y in x)
        )
