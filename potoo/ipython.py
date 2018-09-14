import abc
import base64
import contextlib
from functools import partial
import gc
import os
import random
import signal
import time
from typing import Iterable, Tuple, Union

from attrdict import AttrDict
from dataclasses import dataclass
import IPython
from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import *
from more_itertools import first
import numpy as np
import pandas as pd
import prompt_toolkit

import potoo.pandas
from potoo.pandas import cat_to_str
from potoo.util import AttrContext, or_else, deep_round_sig, singleton


def ipy_format(*xs: any, mimetype='text/plain', join='\n') -> str:
    """
    Format like IPython.display.display
    - Spec: print(ipy_format(*xs)) ~ display(*xs)
    - Manually line-join multiple args like display(x, y) does
    """
    return join.join(
        formats.get(mimetype, formats['text/plain'])
        for x in xs
        for formats in [ipy_all_formats(x)]
    )


def ipy_all_formats(x: any) -> str:
    formats, _metadata = get_ipython().display_formatter.format(x)
    return formats


def ipy_text(*xs: any) -> str:
    return ipy_format(*xs, mimetype='text/plain', join='\n')


def ipy_html(*xs: any) -> str:
    return ipy_format(*xs, mimetype='text/html', join='<br/>')


def ipy_print(*xs: any, **kwargs) -> str:
    """
    Print like IPython.display.display, but also allow control over print kwargs like flush, file, etc.
    - Spec: ipy_print(*xs) ~ display(*xs)
    """
    print(ipy_format(*xs), **kwargs)


def is_ipython_console():
    return or_else(None, lambda: get_ipython().__class__.__name__) == 'TerminalInteractiveShell'


def is_ipython_notebook():
    return or_else(None, lambda: get_ipython().__class__.__name__) == 'ZMQInteractiveShell'


def ipy_load_ext_no_warnings(module_str: str) -> str:
    """Like %load_ext, except silence warnings (e.g. when the extension is already loaded)"""
    # The warnings are emitted from ExtensionMagics.load_ext (see its source), so we just call what it calls
    ipy = get_ipython()
    return ipy.extension_manager.load_extension(module_str)


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
class ipy_formats(AttrContext):
    # A pile of hacks to make values display prettier in ipython/jupyter
    #   - TODO Un-hack these into something that doesn't assume a ~/.pythonrc hook so that ipynb output is reproducible

    deep_round_sig: bool = True  # Very useful by default [and hopefully doesn't break anything...]
    stack_iters: bool = False  # Useful, but maybe not by default

    # Internal state (only used as a stack-structured dynamic var)
    _fancy_cells: bool = False

    @property
    def precision(self):
        return pd.get_option('display.precision')

    # TODO Respect display.max_rows (currently treats it as unlimited)
    def set(self):
        self.ipy = get_ipython()
        if self.ipy:

            # TODO These 'text/plain' formatters:
            #   1. Do nothing, and I don't know why -- maybe interference from potoo.pretty?
            #   2. Are untested, because (1)

            # pd.DataFrame
            self.ipy.display_formatter.formatters['text/html'].for_type(pd.DataFrame, lambda df: (
                self._format_df(df, mimetype='text/html')
            ))
            self.ipy.display_formatter.formatters['text/plain'].for_type(pd.DataFrame, lambda df, p, cycle: (
                p.text(self._format_df(df, mimetype='text/plain'))
            ))

            # pd.Series
            self.ipy.display_formatter.formatters['text/html'].for_type(pd.Series, lambda s: (
                self._format_series(s, mimetype='text/html')
            ))
            self.ipy.display_formatter.formatters['text/plain'].for_type(pd.Series, lambda s, p, cycle: (
                p.text(self._format_series(s, mimetype='text/plain'))
            ))

            # Prevent plotnine plots from displaying their repr str (e.g. '<ggplot: (-9223372036537068975)>') after repr
            # has already side-effected the plotting of an image
            #   - Returning '' causes the formatter to be ignored
            #   - Returning ' ' is a HACK but empirically causes no text output to show up (in atom hydrogen-extras)
            #   - To undo: self.ipy.display_formatter.formatters['text/html'].pop('plotnine.ggplot.ggplot')
            self.ipy.display_formatter.formatters['text/html'].for_type_by_name(
                'plotnine.ggplot', 'ggplot',
                lambda g: ' ',
            )

    def _format_df(self, df: pd.DataFrame, mimetype: str, **df_to_kwargs) -> str:
        with contextlib.ExitStack() as stack:
            # Implicitly trigger _fancy_cells by putting >0 df_cell values in your df (typical usage is whole cols)
            stack.enter_context(self.context(_fancy_cells=df.applymap(lambda x: isinstance(x, df_cell)).any().any()))
            # Format df of cells to a df of formatted strs
            df = df.apply(axis=0, func=lambda col: (col
                # cat_to_str to avoid .apply mapping all cat values, which we don't need and could be slow for large cats
                .pipe(cat_to_str)
                .apply(self._format_df_cell, mimetype=mimetype)
            ))
            # Format df of formatted strs to one formatted str
            if self._fancy_cells:
                # Disable max_colwidth else pandas will truncate and break our strs (e.g. <img> with long data url)
                #   - Isolate this to just df.to_* so that self._format_pd_any (above) sees the real max_colwidth, so
                #     that it correctly renders truncated strs for text/plain cells (via a manual ipy_format)
                #   - TODO Can we clean this up now that we have df_cell_str...?
                stack.enter_context(pd.option_context('display.max_colwidth', -1))
            if mimetype == 'text/html':
                return (
                    df.to_html(**df_to_kwargs,
                        escape=False,  # Allow html in cells
                    )
                    # HACK Hide '\n' from df.to_html, else it incorrectly renders them as '\\n' (and breaks <script>)
                    .replace('\a', '\n')
                )
            else:
                return df.to_string(**df_to_kwargs)

    def _format_series(self, s: pd.Series, mimetype: str) -> str:
        # cat_to_str to avoid .apply mapping all cat values, which we don't need and could be slow for large cats
        text = s.pipe(cat_to_str).apply(self._format_pd_any, mimetype=mimetype).to_string()
        if mimetype == 'text/html':
            # df_cell as <pre> instead of fancy html since Series doesn't properly support .to_html like DataFrames do
            #   - https://github.com/pandas-dev/pandas/issues/8829
            #   - Use <div style=...> instad of <pre>, since <pre> brings along a lot of style baggage we don't want
            return '<div style="white-space: pre">%s</div>' % text
        else:
            return text

    def _format_df_cell(self, x: any, mimetype: str) -> any:

        # We exclude dicts/mappings here since silently showing only dict keys (because iter(dict)) would be confusing
        #   - In most cases it's preferred to apply df_cell_stack locally instead of setting stack_iters=True globally,
        #     but it's important to keep the global option in cases like %%sql magic, where the result df displays as is
        if self.stack_iters and isinstance(x, (list, tuple, np.ndarray)):
            x = df_cell_stack(x)

        ret = self._format_pd_any(x, mimetype=mimetype)

        # HACK An ad-hoc, weird thing to help out atom/jupyter styling
        #   - TODO Should this be: _has_number(ret)? _has_number(x)? _has_number(col) from one frame above?
        if mimetype == 'text/html' and not self._has_number(ret):
            ret = '<div class="not-number">%s</div>' % (ret,)

        # HACK Hide '\n' from df.to_html, else it incorrectly renders them as '\\n' (and breaks <script>)
        if mimetype == 'text/html' and isinstance(ret, str):
            ret = ret.replace('\n', '\a')

        return ret

    def _format_pd_any(self, x: any, mimetype: str) -> any:
        # HACK HACK HACK Way too much stuff going on in here that's none of our business...

        # If not _fancy_cells, defer formatting to pandas (to respect e.g. 'display.max_colwidth')
        #   - Principle of least surprise: if I put no df_cell's in my df, everything should be normal
        if not self._fancy_cells:
            # Apply self.precision to numbers, like numpy but everywhere
            #   - But only if deep_round_sig
            #   - And don't touch np.array's since they can be really huge, and numpy already truncates them for us
            #   - TODO How to achieve self.precision less brutishly?
            if self.deep_round_sig and not isinstance(x, np.ndarray):
                return deep_round_sig(x, self.precision)
            else:
                return x

        # If _fancy_cells but not a df_cell value, manually emulate pandas formatting
        #   - This is necessary only because we have to disable 'display.max_colwidth' above to accommodate long
        #     formatted strs (e.g. <img>) from df_cell values (next condition)
        #   - This emulation will violate the principle of least surprise if do something wrong, which is why take care
        #     to avoid it if the user put no df_cell's in their df (via _fancy_cells)
        #   - TODO What are we missing by not reusing pd.io.formats.format.format_array?
        elif not isinstance(x, df_cell):
            # Do the unfancy conversion (e.g. for self.precision)
            with self.context(_fancy_cells=False):
                x = self._format_pd_any(x, mimetype=mimetype)
            # Truncate str(x) to 'display.max_colwidth' _only if necessary_, else leave x as is so pandas can format it
            #   - e.g. datetime.date: pandas '2000-01-01' vs. str() 'datetime.date(2000, 1, 1)'
            return truncate_like_pd_max_colwidth(x)

        # If _fancy_cells and a df_cell value, format to html str like ipython display()
        else:
            return ipy_formats_to_mimetype(x, mimetype=mimetype)

    def _has_number(self, x: any) -> bool:
        return (
            np.issubdtype(type(x), np.number) or
            isinstance(x, list) and any(self._has_number(y) for y in x)
        )

    # XXX Subsumed by deep_round_sig
    #
    # def _format_pd_any(self, x: any) -> any:
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


def ipy_formats_to_text(x: any) -> str:
    """Format to text str using ipython formatters, to emulate ipython display()"""
    return ipy_formats_to_mimetype(x, mimetype='text/plain')


def ipy_formats_to_html(x: any) -> str:
    """Format to html str using ipython formatters, to emulate ipython display()"""
    return ipy_formats_to_mimetype(x, mimetype='text/html')


def ipy_formats_to_mimetype(x: any, mimetype: str) -> str:
    """Format to str for mimetype using ipython formatters, to emulate ipython display()"""
    # TODO Clean up / modularize dispatch on format
    # TODO Handle more formats (svg, ...)
    ret = None
    formats = ipy_all_formats(x)
    if mimetype == 'text/html':
        html = formats.get('text/html')
        img_types = [k for k in formats.keys() if k.startswith('image/')]
        if html:
            ret = html.strip()
        elif img_types:
            img_type = first(img_types)
            img = formats[img_type]
            if isinstance(img, str):
                img_b64 = img.strip()
            elif isinstance(img, bytes):
                img_b64 = base64.b64encode(img).decode()
            ret = f'<img src="data:{img_type};base64,{img_b64}"></img>'
    if ret is None:
        ret = formats['text/plain']
    return ret


@dataclass
class df_cell:
    """
    Mark a df cell to be displayed in some special way determined by the subtype
    - Motivating use case is ipy_formats._format_df
    """

    value: any

    @abc.abstractmethod
    def _repr_mimebundle_(self, include=None, exclude=None):
        ...

    @classmethod
    def many(cls, xs: Iterable[any]) -> Iterable['cls']:
        return [cls(x) for x in xs]


# TODO Untested
class df_cell_union(df_cell):
    """
    Union the mimebundles of multiple df_cell's
    """

    def _repr_mimebundle_(self, include=None, exclude=None):
        df_cells = list(self.value)  # Materialize iters
        df_cells = [x if isinstance(x, df_cell) else df_cell_display(x) for x in df_cells]  # Coerce to df_cell
        ret = {}
        for cell in df_cells:
            for k, v in cell._repr_mimebundle_(include=include, exclude=exclude).items():
                print(('[kv]', k, v))
                ret.setdefault(k, v)
        return ret


class df_cell_display(df_cell):
    """
    Mark a df cell to be displayed like ipython display()
    """

    def _repr_mimebundle_(self, include=None, exclude=None):
        # Ref: https://ipython.readthedocs.io/en/stable/config/integrating.html
        formats, _metadata = get_ipython().display_formatter.format(self.value)
        return formats


class df_cell_str(df_cell):
    """
    Mark a df cell to be displayed like str(value)
    - Useful e.g. for bypassing pandas 'display.max_colwidth' or ipython str wrapping
    """

    def _repr_mimebundle_(self, include=None, exclude=None):
        return {
            'text/plain': str(self.value),
            'text/html': str(self.value),
        }


class df_cell_stack(df_cell):
    """
    Mark a df cell to be displayed as vertically stacked values (inspired by the bigquery web UI)
    - Assumes the cell value is iterable
    """

    def _repr_mimebundle_(self, include=None, exclude=None):
        self.value = list(self.value)  # Materialize iters
        return {
            'text/plain': '' if len(self.value) == 0 else (
                pd.Series(ipy_formats._format_pd_any(x, mimetype='text/plain') for x in self.value)
                .to_string(index=False)
            ),
            'text/html': '' if len(self.value) == 0 else (
                pd.Series(ipy_formats._format_pd_any(x, mimetype='text/html') for x in self.value)
                .to_string(index=False)
                .replace('\n', '<br/>')
            ),
        }


def truncate_like_pd_max_colwidth(x: any) -> str:
    """
    Emulate the behavior of pandas 'display.max_colwidth'
    - TODO Ugh, how can we avoid doing this ourselves?
    """
    max_colwidth = pd.get_option("display.max_colwidth")
    if max_colwidth is None:
        return x
    else:
        s = str(x)
        if len(s) <= max_colwidth:
            return s
        else:
            return s[:max_colwidth - 3] + '...'


def ipy_install_mock_get_ipython():
    """
    Ever wanted to use get_ipython() from a normal python process? Like a production webserver? Here you go.

    This is pretty heinous, but it works great!
    """

    # Create a mock ipy
    ipy = IPython.terminal.embed.InteractiveShellEmbed()
    ipy.dummy_mode = True  # Not sure what this does, but sounds maybe really important

    # "Enable all formatters", else you only get 'text/plain', never 'text/html' (or others)
    #   - From: https://github.com/ipython/ipython/blob/0de2f49/IPython/core/tests/test_interactiveshell.py#L779-L780
    ipy.display_formatter.active_types = ipy.display_formatter.format_types

    # Moneypatch get_ipython() to return our mock ipy
    InteractiveShell.initialized = lambda *args, **kwargs: True
    InteractiveShell.instance = lambda *args, **kwargs: ipy
    assert get_ipython() is not None

    # Tests:
    #   df = pd.DataFrame(['foo'])
    #   assert 'text/html' in ipy.display_formatter.format(df)[0]
