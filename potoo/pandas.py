from collections import OrderedDict
from contextlib import contextmanager
from functools import partial, wraps
import os
import signal
import subprocess
import sys
import types
from typing import Callable, Iterable, List, Optional, Tuple, Union

import humanize
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

import potoo.numpy
from potoo.util import get_cols, get_rows, or_else

# Convenient shorthands for interactive use, but not recommended for real code that needs to be readable and maintanable
DF = pd.DataFrame
S = pd.Series

# Mutate these for manual control
#   - https://pandas.pydata.org/pandas-docs/stable/options.html
#   - TODO In ipykernel you have to manually set_display() after changing any of these
#       - Workaround: use pd.set_option for the display_* settings
ipykernel_display_max_rows = 1000   # For pd df output
ipykernel_display_width    = 10000  # For pd df output
ipykernel_lines            = 75     # Does this affect anything?
ipykernel_columns          = 120    # For ipython pretty printing (not dfs)
display_width              = 0  # Default: 80; 0 means use get_terminal_size, ''/None means unlimited
display_max_rows           = 0  # Default: 60; 0 means use get_terminal_size, ''/None means unlimited
display_max_columns        = 250  # Default: 20
display_max_colwidth       = lambda cols: 200  # Default: 50; go big for dense bq cells
display_precision          = 3  # Default: 6; better magic than _float_format


def set_display_max_colwidth(x=display_max_colwidth):
    global display_max_colwidth
    if isinstance(x, types.FunctionType):
        display_max_colwidth = x
    elif isinstance(x, float):
        display_max_colwidth = lambda cols: int(cols * x)
    elif isinstance(x, int):
        display_max_colwidth = lambda cols: x
    return display_max_colwidth


def set_display_precision(x=display_precision):
    global display_precision
    display_precision = x
    return display_precision


def set_display(**kwargs):
    "Make everything nice"

    # XXX I couldn't find a way to make auto-detect work with both ipython (terminal) + ipykernel (atom)
    # # Unset $LINES + $COLUMNS so pandas will detect changes in terminal size after process start
    # #   - https://github.com/pandas-dev/pandas/blob/473a7f3/pandas/io/formats/terminal.py#L32-L33
    # #   - https://github.com/python/cpython/blob/7028e59/Lib/shutil.py#L1071-L1079
    # #   - TODO These used to be '' instead of del. Revert back if this change causes problems.
    # os.environ.pop('LINES', None)
    # os.environ.pop('COLUMNS', None)

    # HACK This is all horrible and I hate it. After much trial and error I settled on this as a way to make both
    # ipython (terminal) and ipykernel (atom) work.
    try:
        size = os.get_terminal_size(sys.__stdout__.fileno())
    except OSError:
        # If ipykernel
        lines = ipykernel_lines
        columns = ipykernel_columns
        _display_width = display_width or ipykernel_display_width or columns
        _display_max_rows = display_max_rows or ipykernel_display_max_rows or lines
    else:
        # If terminal
        lines = size.lines - 8
        columns = size.columns
        _display_width = display_width or columns
        _display_max_rows = display_max_rows or lines

    # Let kwargs override any of these params that we just inferred
    lines = kwargs.get('lines', lines)
    columns = kwargs.get('columns', columns)
    _display_width = kwargs.get('_display_width', _display_width)
    _display_max_rows = kwargs.get('_display_max_rows', _display_max_rows)

    # For ipython pretty printing (not dfs)
    os.environ['LINES'] = str(lines)
    os.environ['COLUMNS'] = str(columns)

    potoo.numpy.set_display()

    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.set_option.html
    #   - TODO Any good way to %page by default?
    #       - here: pd.set_option('display.width', 10000)
    #       - repl: pd.DataFrame({i:range(100) for i in range(100)})
    pd.set_option('display.width',        _display_width)
    pd.set_option('display.max_rows',     _display_max_rows)
    pd.set_option('display.max_columns',  display_max_columns)
    pd.set_option('display.max_colwidth', display_max_colwidth(get_cols()))
    pd.set_option('display.precision',    display_precision)  # Default: 6; better magic than _float_format
    # pd.set_option('display._float_format', _float_format(10, 3))  # Default: magic in pandas.formats.format


def set_display_on_sigwinch():
    "set_display on window change (SIGWINCH)"
    signal.signal(signal.SIGWINCH, lambda sig, frame: set_display())
    set_display()  # And ensure it's set to begin with


# TODO Check out `with pd.option_context`
@contextmanager
def with_options(options):
    saved = {}
    for k, v in options.items():
        saved[k] = pd.get_option(k)
        pd.set_option(k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            pd.set_option(k, v)


# Based on https://github.com/pandas-dev/pandas/issues/8517#issuecomment-247785821
def df_flatmap(df: pd.DataFrame, f: Callable[['Row'], Union[pd.DataFrame, Iterable['Row']]]) -> pd.DataFrame:
    return pd.DataFrame(
        OrderedDict(row_out)
        for _, row_in in df.iterrows()
        for f_out in [f(row_in)]
        for row_out in (
            (row_out for i, row_out in f_out.iterrows()) if isinstance(f_out, pd.DataFrame) else
            f_out
        )
    )


def df_summary(
    # A df, or a series that will be coerced into a 1-col df
    df: Union[pd.DataFrame, pd.Series],
    # Summaries that might have a different dtype than the column they summarize (e.g. count, mean)
    stats=[
        # Use dtype.name (str) instead of dtype (complicated object that causes trouble)
        ('dtype', lambda df: [dtype.name for dtype in df.dtypes]),
        # ('sizeof', lambda df: _sizeof_df_cols(df).map(partial(humanize.naturalsize, binary=True))),
        ('sizeof', lambda df: _sizeof_df_cols(df)),
        ('len', lambda df: len(df)),
        'count',
        # df.apply + or_else to handle unhashable types
        ('nunique', lambda df: df.apply(lambda c: or_else(np.nan, lambda: c.nunique()))),
        # df.apply + or_else these else they subset the cols to just the numerics, which quietly messes up col ordering
        #   - reduce=False else all dtypes are 'object' [https://stackoverflow.com/a/34917685/397334]
        #   - dtype.base else 'category' dtypes break np.issubdtype [https://github.com/pandas-dev/pandas/issues/9581]
        ('mean', lambda df: df.apply(reduce=False, func=lambda c: c.mean() if np.issubdtype(c.dtype.base, np.number) else np.nan)),
        ('std',  lambda df: df.apply(reduce=False, func=lambda c: c.std()  if np.issubdtype(c.dtype.base, np.number) else np.nan)),
    ],
    # Summaries that have the same dtype as the column they summarize (e.g. quantile values)
    prototypes=[
        ('min', lambda df: _df_quantile(df, .0,  interpolation='lower')),
        ('25%', lambda df: _df_quantile(df, .25, interpolation='lower')),
        ('50%', lambda df: _df_quantile(df, .5,  interpolation='lower')),
        ('75%', lambda df: _df_quantile(df, .75, interpolation='lower')),
        ('max', lambda df: _df_quantile(df, 1,   interpolation='higher')),
    ],
):
    """A more flexible version of df.describe, with more information by default"""
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)
    stats = [(f, lambda df, f=f: getattr(df, f)()) if isinstance(f, str) else f for f in stats]
    prototypes = [(f, lambda df, f=f: getattr(df, f)()) if isinstance(f, str) else f for f in prototypes]
    return (
        pd.DataFrame(OrderedDict({k: f(df) for k, f in stats + prototypes})).T
        # Reorder cols to match input (some aggs like mean/std throw out non-numeric cols, which messes up order)
        [df.columns]
        # Pull stats up into col index, so that our col dtypes can match the input col dtypes
        .T.set_index([k for k, f in stats], append=True).T
    )


def _df_quantile(df, q=.5, interpolation='linear'):
    """Like pd.DataFrame.quantile but handles ordered categoricals"""
    return df.apply(
        lambda c: _series_quantile(c, q=q, interpolation=interpolation),
        reduce=False,  # Else all dtypes are 'object' [https://stackoverflow.com/a/34917685/397334]
    )

def _series_quantile(s, *args, **kwargs):
    """Like pd.Series.quantile but handles ordered categoricals"""
    if s.dtype.name == 'category':
        cat_code = s.cat.codes.quantile(*args, **kwargs)
        return s.dtype.categories[cat_code] if cat_code != -1 else None
    else:
        try:
            return s.quantile(*args, **kwargs)
        except:
            # e.g. a column of non-uniform np.array's will fail like:
            #   ValueError: operands could not be broadcast together with shapes (6599624,) (459648,)
            return np.nan


def _sizeof_df_cols(df: pd.DataFrame) -> 'Column[int]':
    """
    sizeof is hard, but make our best effort:
    - Use dask.sizeof.sizeof instead of sys.getsizeof, since the latter is unreliable for pandas/numpy objects
    - Use df.applymap, since dask.sizeof.sizeof appears to not do this right [why? seems wrong...]
    """
    try:
        import dask.sizeof
    except:
        return df.apply(lambda c: None)
    else:
        return df.applymap(dask.sizeof.sizeof).sum()


def df_reorder_cols(df: pd.DataFrame, first: List[str] = [], last: List[str] = []) -> pd.DataFrame:
    first_last = set(first) | set(last)
    return df.reindex(columns=list(first) + [c for c in df.columns if c not in first_last] + list(last))


def df_transform_columns(df: pd.DataFrame, f: Callable[[List[str]], List[str]]) -> pd.DataFrame:
    df = df.copy()
    df.columns = f(df.columns)
    return df


def df_transform_column_names(df: pd.DataFrame, f: Callable[[str], str]) -> pd.DataFrame:
    return df_transform_columns(df, lambda cs: [f(c) for c in df.columns])


def df_transform_index(df: pd.DataFrame, f: Callable[[List[str]], List[str]]) -> pd.DataFrame:
    df = df.copy()
    df.index = f(df.index)
    return df


def df_remove_unused_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Do col.remove_unused_categories() for all categorical columns
    """
    return (df.assign(**{
        k: df[k].cat.remove_unused_categories()
        for k in df.columns
        if df[k].dtype.name == 'category'
    }))


def df_ordered_cat(df: pd.DataFrame, *args, transform=lambda x: x, **kwargs) -> pd.DataFrame:
    """
    Map many str series to ordered category series
    """
    cats = dict(
        **{k: lambda df: df[k].unique() for k in args},
        **kwargs,
    )
    return (df.assign(**{
        k: as_ordered_cat(df[k], list(transform(
            x(df) if isinstance(x, types.FunctionType) else x
        )))
        for k, x in cats.items()
    }))


def as_ordered_cat(s: pd.Series, ordered_cats: List[str] = None) -> pd.Series:
    """
    Map a str series to an ordered category series
    - If ordered_cats isn't given, list(s) is used (which must produce unique values)
    """
    return s.astype(CategoricalDtype(ordered_cats or list(s), ordered=True))


def df_cat_to_str(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map any categorical columns to str columns (see cat_to_str for details)
    """
    return df.apply(cat_to_str, axis=0)


def cat_to_str(s: pd.Series) -> pd.Series:
    """
    If s is a category dtype, map it to a str. This is useful when you want to avoid bottlenecks on large cats:
    - s.apply(f) will apply f to each value in s _and_ each value in the category, to make the new output category dtype
    - cat_to_str(s).apply(f) will apply f only to each value in s, since there's no output category dtype to compute
    """
    return s.astype('str') if s.dtype.name == 'category' else s


def df_transform_cat(df: pd.DataFrame, f: Callable[[List[str]], Iterable[str]], *col_names) -> pd.DataFrame:
    """
    Transform the cat.categories values to f(cat.categories) for each category column given in col_names
    """
    return df.assign(**{col_name: transform_cat(df[col_name], f) for col_name in col_names})


def df_reverse_cat(df: pd.DataFrame, *col_names) -> pd.DataFrame:
    """
    Reverse the cat.categories values of each (ordered) category column given in col_names
    - Useful e.g. for reversing plotnine axes: https://github.com/has2k1/plotnine/issues/116#issuecomment-365911195
    """
    return df_transform_cat(df, reversed, *col_names)


def transform_cat(s: pd.Series, f: Callable[[List[str]], Iterable[str]]) -> pd.Series:
    """
    Transform the category values of a categorical series
    """
    return s.astype('str').astype(CategoricalDtype(
        categories=list(f(s.dtype.categories)),
        ordered=s.dtype.ordered,
    ))


def reverse_cat(s: pd.Series) -> pd.Series:
    """
    Reverse the category values of a categorical series
    - Useful e.g. for reversing plotnine axes: https://github.com/has2k1/plotnine/issues/116#issuecomment-365911195
    """
    return transform_cat(s, reversed)


def df_ensure(df, **kwargs):
    """
    df.assign only the columns that aren't already present
    """
    return df.assign(**{
        k: v
        for k, v in kwargs.items()
        if k not in df
    })
    return df


# XXX Obviated by df_ensure?
# def produces_cols(*cols):
#     cols = [c for c in cols if c != ...]
#     def decorator(f):
#         @wraps(f)
#         def g(*args, **kwargs) -> pd.DataFrame:
#             df = _find_df_in_args(*args, **kwargs)
#             _refresh = kwargs.pop('_refresh', False)
#             if _refresh or not cols or any(c not in df for c in cols):
#                 df = f(*args, **kwargs)
#             return df
#         return g
#     return decorator


def requires_cols(*cols):
    cols = [c for c in cols if c != ...]
    def decorator(f):
        @wraps(f)
        def g(*args, **kwargs) -> any:
            df = _find_df_in_args(*args, **kwargs)
            if any(c not in df for c in cols):
                raise ValueError(f'Expected cols[{cols}] in df.columns[{df.columns}]')
            return f(*args, **kwargs)
        return g
    return decorator


def _find_df_in_args(*args, **kwargs):
    for x in [*args, *kwargs.values()]:
        if isinstance(x, pd.DataFrame):
            return x
    else:
        raise ValueError('No df found in args')


def df_style_cell(*styles: Union[
    Tuple[Callable[['cell'], bool], 'style'],
    Tuple['cell', 'style'],
    Callable[['cell'], Optional['style']],
]) -> Callable[['cell'], 'style']:
    """
    Shorthand for df.style.applymap(...). Example usage:
        df.style.applymap(df_style_cell(
            (lambda x: 0 < x < 1, 'color: red'),
            (0, 'color: green'),
            lambda x: 'background: %s' % to_rgb_hex(x),
        ))
    """
    def f(x):
        y = None
        for style in styles:
            if isinstance(style, tuple) and isinstance(style[0], types.FunctionType) and style[0](x):
                y = style[1]
            elif isinstance(style, tuple) and x == style[0]:
                y = style[1]
            elif isinstance(style, types.FunctionType):
                y = style(x)
            if y:
                break
        return y or ''
    return f


#
# sql/bq
#


# TODO What's the right way to manage sessions and txns?
def pd_read_sql(session, sql):
    session.rollback()
    try:
        return pd.read_sql(sql, session.connection())
    finally:
        session.rollback()


# TODO -> potoo.sqlalchemy
def raw_sql(session, sql):
    return (dict(x.items()) for x in session.execute(sql))


def pd_read_bq(
    query,
    project_id=None,
    dialect='standard',
    # read_gbq=pd.io.gbq.read_gbq,                   # Sequential IO, slow
    # read_gbq=potoo.pandas_io_gbq_par_io.read_gbq,  # Parallel IO (via dask), ballpark ~4x faster than sequential IO
    read_gbq=None,                                   # Lazily loads potoo.pandas_io_gbq_par_io.read_gbq
    **kwargs
):
    """
    Example usage:
        df = pd_read_bq('''
            select ...
            from ...
        ''')

    Docs:
    - http://pandas.pydata.org/pandas-docs/stable/generated/pandas.io.gbq.read_gbq.html
    - https://cloud.google.com/bigquery/docs/
    """

    if read_gbq is None:
        import potoo.pandas_io_gbq_par_io
        read_gbq = potoo.pandas_io_gbq_par_io.read_gbq

    return read_gbq(
        query=query,
        dialect=dialect,
        project_id=project_id or bq_default_project(),
        **kwargs
    )


def bq_default_project():
    return subprocess.check_output(
        'gcloud config get-value project 2>/dev/null',
        shell=True,
    ).decode('utf8').strip()
