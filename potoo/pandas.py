from contextlib import contextmanager
import os
import signal
import subprocess
import sys
import types

import pandas as pd

import potoo.numpy
# from potoo.numpy import _float_format
from potoo.util import get_cols, get_rows


# Mutate these for manual control
#   - TODO In ipykernel you have to manually set_display() after changing any of these
#       - Workaround: use pd.set_option for the display_*)
ipykernel_display_max_rows = 1000   # For pd df output
ipykernel_display_width    = 10000  # For pd df output
ipykernel_lines            = 75     # Does this affect anything?
ipykernel_columns          = 120    # For ipython pretty printing (not dfs)
display_width              = 0  # Default: 80; 0 means use get_terminal_size, ''/None means unlimited
display_max_rows           = 0  # Default: 60; 0 means use get_terminal_size, ''/None means unlimited
display_max_columns        = None  # Default: 20
display_max_colwidth       = lambda cols: 10000  # Default: 50; go big for dense bq cells
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


def set_display():
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


def pd_flatmap(df, f):
    return pd.DataFrame.from_records(
        y
        for x in df.itertuples(index=False)
        for y in f(x)
    )


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
