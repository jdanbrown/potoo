from collections import OrderedDict
from contextlib import contextmanager
from functools import partial, wraps
import os
import signal
import subprocess
import sys
import types
from typing import Callable, Iterable, Iterator, List, Mapping, Optional, Tuple, TypeVar, Union

import humanize
from more_itertools import flatten, one, unique_everseen, windowed
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

import potoo.numpy
from potoo.util import get_cols, get_rows, or_else

# Convenient shorthands for interactive use -- not recommended for durable code that needs to be read and maintained
DF = pd.DataFrame
S = pd.Series

X = TypeVar('X')


#
# Global options
#

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


#
# Utils
#


def df_rows(df) -> Iterator['Row']:
    """Shorthand for a very common idiom"""
    return (row for i, row in df.iterrows())


def df_map_rows(df, f: Callable[['Row'], 'Row'], *args, **kwargs) -> pd.DataFrame:
    """Shorthand for a very common idiom"""
    return df.apply(axis=1, func=f, *args, **kwargs)


def series_assign(s: pd.Series, **kwargs) -> pd.Series:
    """Like df.assign but for Series"""
    s = s.copy()
    for k, v in kwargs.items():
        s.at[k] = v if not callable(v) else v(s.at[k])
    return s


def df_assign_first(df, **kwargs) -> pd.DataFrame:
    """Like df.assign but also reposition the assigned cols to be first"""
    return (df
        .assign(**kwargs)
        .pipe(df_reorder_cols, first=kwargs.keys())
    )


def df_col_map(df, **kwargs) -> pd.DataFrame:
    """
    Map col values by the given function
    - A shorthand for a very common usage of df.assign / df.col.map
    """
    return df.assign(**{
        c: df[c].map(f)
        for c, f in kwargs.items()
    })


# Based on https://github.com/pandas-dev/pandas/issues/8517#issuecomment-247785821
#   - Not very performant, use sparingly...
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
    try:
        df = df.reset_index()  # Surface indexes as cols in stats
    except:
        # Oops, index is already a col [`drop=df.index.name in df.columns` is unreliable b/c df.index.names ...]
        df = df.reset_index(drop=True)
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
    return df.memory_usage(index=False, deep=True)


# XXX Looks like df.memory_usage(deep=True) is more accurate (previous attempts were missing deep=True)
# def _sizeof_df_cols(df: pd.DataFrame) -> 'Column[int]':
#     """
#     sizeof is hard, but make our best effort:
#     - Use dask.sizeof.sizeof instead of sys.getsizeof, since the latter is unreliable for pandas/numpy objects
#     - Use df.applymap, since dask.sizeof.sizeof appears to not do this right [why? seems wrong...]
#     """
#     try:
#         import dask.sizeof
#     except:
#         return df.apply(lambda c: None)
#     else:
#         return df.applymap(dask.sizeof.sizeof).sum()


def df_value_counts(
    df: pd.DataFrame,
    exprs=None,          # Cols to surface, as expressions understood by df.eval(expr) (default: df.columns)
    limit=10,            # Limit rows
    exclude_max_n=1,     # Exclude cols where max n ≤ exclude_max_n
    fillna='',           # Fill na cells (for seeing); pass None to leave na cols as NaN (for processing)
    unique_names=False,  # Give all cols unique names (for processing) instead of reusing 'n' (for seeing)
    **kwargs,            # kwargs for .value_counts (e.g. dropna)
) -> pd.DataFrame:
    """Series.value_counts() extended over a whole DataFrame (with a few compromises in hygiene)"""
    exprs = exprs if exprs is not None else df.columns
    return (df
        .pipe(df_remove_unused_categories)
        .pipe(df_cat_to_str)
        .pipe(lambda df: (pd.concat(axis=1, objs=[
            ns
            for expr_opts in exprs
            for expr, opts in [expr_opts if isinstance(expr_opts, tuple) else (expr_opts, dict())]
            for ns in [(df
                .eval(expr)
                .value_counts(**kwargs)
            )]
            if ns.iloc[0] > exclude_max_n
            for ns in [(ns
                .pipe(lambda s: (
                    # NOTE We "sort_index" when "sort_values=True" because the "values" are in the index, as opposed to
                    # the "counts", which are the default sort
                    s.sort_values(ascending=opts.get('ascending', False)) if not opts.get('sort_values') else
                    s.sort_index(ascending=opts.get('ascending', True))
                ))
                .iloc[:limit]
                .to_frame()
                .rename(columns=lambda x: f'n_{expr}' if unique_names else 'n')
                .reset_index()
                .rename(columns={'index': expr})
            )]
        ])))
        .fillna(fillna)
    )


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


def df_set_index_name(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return df_transform_index(df, lambda index: index.rename(name))


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


def requires_cols(*required_cols):
    required_cols = [c for c in required_cols if c != ...]
    def decorator(f):
        @wraps(f)
        def g(*args, **kwargs) -> any:
            input = _find_first_df_or_series_in_args(*args, **kwargs)
            input_cols = input.columns if isinstance(input, pd.DataFrame) else input.index  # df.columns or series.index
            if not set(required_cols) <= set(input_cols):
                raise ValueError(f'requires_col: required_cols[{required_cols}] not all in input_cols[{input_cols}]')
            return f(*args, **kwargs)
        return g
    return decorator


def _find_first_df_or_series_in_args(*args, **kwargs):
    for x in [*args, *kwargs.values()]:
        if isinstance(x, (pd.DataFrame, pd.Series)):
            return x
    else:
        raise ValueError('No df or series found in args')


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
# io
#

def pd_read_fwf(
    filepath_or_buffer,
    widths: Optional[Union[List[int], 'infer']] = None,
    unused_char='\a',  # For widths='infer': any char not present in the file, used to initially parse raw lines
    **kwargs,
) -> pd.DataFrame:
    """
    Like pd.read_fwf, except:
    - Add support for widths='infer', which infers col widths from the header row (assuming no spaces in header names)
    """

    if widths == 'infer':
        # Read raw lines
        #   - Use pd.read_* (with kwargs) so we get all the file/str/compression/encoding goodies
        [header_line, *body_lines] = (
            pd.read_csv(filepath_or_buffer, **kwargs, header=None, sep=unused_char)
            .pipe(lambda df: map(one, df.to_records(index=False)))
        )
        [header_line, *body_lines]
        # Compute col widths
        #   - header_line determines widths for all but the last col, which might have values that extend past the header
        #   - Incorporate body_lines to determine the width of the last col
        widths_cum = [
            i + 1
            for (i, c), (_, d) in windowed(enumerate(header_line), 2)
            if c != ' ' and d == ' '
        ]
        widths_cum = [
            0,
            *widths_cum,
            max(len(line) for line in [header_line, *body_lines]),
        ]
        widths = [
            y - x
            for x, y in windowed(widths_cum, 2)
        ]

    return pd.read_fwf(filepath_or_buffer,
        widths=widths,
        **kwargs,
    )


def df_to_fwf_df(df: pd.DataFrame, reset_index=True, fresh_col_name='_index') -> pd.DataFrame:
    """
    Transform a df so that its .to_string() is copy/pastable to a .fwf format
    - HACK This function returns a new df that ipython/jupyter will display in a copy/paste-able form
    - TODO Make a function df_to_fwf that actually does the real df->str/file
    """
    assert fresh_col_name not in df.columns, f"Refusing to overwrite your col named {fresh_col_name!r}"
    if df.index.name is not None and reset_index:
        df = df.reset_index()
    return (df
        .assign(**{fresh_col_name: ''})
        .set_index(fresh_col_name)
        .pipe(df_set_index_name, None)
        # Cosmetic: ensure 2 spaces between columns in .to_string()
        #   - The behavior of .to_string() is 2 spaces of margin around cells, but only 1 space of margin around headers
        #   - Padding each header string with 1 (leading) space effects 2 spaces of margin around both cells and headers
        #   - This isn't necessary for pd.read_fwf to work, but it's helpful to the human interacting with the file
        .rename(columns=lambda c: ' ' + c)
    )


#
# "Plotting", i.e. styling df html via mpl/plotnine color palettes
#


# TODO Add df_color_col for continuous values (this one is just for discrete values)
def df_col_color_d(
    df,
    _join=',',
    _stack=False,
    _extend_cmap=False,
    **col_cmaps,
) -> pd.DataFrame:
    """Color the (discrete) values in a df column (like plotnine.scale_color_cmap_d for tables)"""

    # Lazy imports so we don't hard-require these heavy-ish libs
    from IPython.display import HTML
    from mizani.palettes import cmap_d_pal
    from potoo.plot import mpl_cmap_repeat

    # Break cycling import
    from potoo.ipython import df_cell_display, df_cell_stack

    def iter_or_singleton(x: Union[Iterable[X], X]) -> Iterable[X]:
        return [x] if not hasattr(x, '__len__') or isinstance(x, str) else x
    def color_col(s: pd.Series, cmap):
        s = cat_to_str(s)  # Else iter_or_singleton tries to make a category of lists, which barfs when it tries to hash
        vs = list(unique_everseen(
            v
            for v in flatten(s.map(iter_or_singleton))
            if pd.notnull(v)
        ))
        # TODO Allow user to control this ordering, like plotnine allows with category dtypes
        vs = sorted(vs)
        if _extend_cmap:
            cmap = mpl_cmap_repeat(len(vs), cmap)
        colors = dict(zip(vs, cmap_d_pal(cmap)(len(vs))))
        # FIXME 'text/plain' gets '' from HTML(...) [repro-able? why does this happen?]
        join = lambda xs: df_cell_stack(xs) if _stack else df_cell_display(HTML(_join.join(xs)))
        return s.apply(lambda v: join(
            _html_color(v, colors)
            for v in iter_or_singleton(v)
        ))
    return df.assign(**{
        col: color_col(df[col], cmap)
        for col, cmap in col_cmaps.items()
    })


def df_cell_color(x: any, colors: Mapping[any, str]) -> 'df_cell':
    from potoo.ipython import df_cell_display  # Break cycling import
    return df_cell_display(HTML(_html_color(x, colors)))


def _html_color(x: any, colors: Mapping[any, str]) -> str:
    return '<span style="color: %s">%s</span>' % (colors.get(x, 'inherit'), x)


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
