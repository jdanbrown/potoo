from functools import partial

from potoo.util import get_cols


# Use ipy pretty if we have it
try:
    import IPython.lib.pretty as _pp
except:
    pass
else:
    pp      = partial(_pp.pprint, max_width=get_cols())
    pformat = partial(_pp.pretty, max_width=get_cols())


# Override with pp-ez if we have it
try:
    import pp as _pp
except:
    pass
else:
    pp      = partial(_pp,         width=get_cols(), indent=2)
    pformat = partial(_pp.pformat, width=get_cols(), indent=2)
