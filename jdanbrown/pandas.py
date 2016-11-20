import pandas as pd

import jdanbrown.numpy
# from jdanbrown.numpy import _float_format
from jdanbrown.util import get_rows, get_cols


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


# TODO -> jdanbrown.sqlalchemy
def raw_sql(session, sql):
    return (dict(x.items()) for x in session.execute(sql))


display_width       = lambda cols: cols
display_max_rows    = lambda rows: rows - 7
display_max_columns = 100000
display_precision   = 3


def set_display():

    jdanbrown.numpy.set_display()

    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.set_option.html
    #   - TODO Any good way to %page by default?
    #       - here: pd.set_option('display.width', 10000)
    #       - repl: pd.DataFrame({i:range(100) for i in range(100)})
    pd.set_option('display.width',        max(1, int(display_width(get_cols()))))     # Default: 80
    pd.set_option('display.max_rows',     max(1, int(display_max_rows(get_rows()))))  # Default: 60
    pd.set_option('display.max_columns',  display_max_columns)                        # Default: 20
    pd.set_option('display.precision',    display_precision)  # Default: 6; better magic than _float_format
    # pd.set_option('display._float_format', _float_format(10, 3))  # Default: magic in pandas.formats.format
