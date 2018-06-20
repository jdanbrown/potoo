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


# TODO Figure this out, looks more flexible that pp (e.g. can register 3rd-party types)
#   - Why are dataclasses printing with no fields? -- the dataclasses extra appears to be installed...
#       - https://github.com/tommikaikkonen/prettyprinter/blob/master/prettyprinter/extras/dataclasses.py
#   - Figure out if we want all of 'python' / 'ipython' / 'ipython_repr_pretty' by default
#       - https://prettyprinter.readthedocs.io/en/latest/api_reference.html#prettyprinter.install_extras
#   - Docs:
#       - https://prettyprinter.readthedocs.io/en/latest/
#       - https://prettyprinter.readthedocs.io/en/latest/usage.html
#       - https://prettyprinter.readthedocs.io/en/latest/api_reference.html
# Override with prettyprinter if we have it
try:
    import prettyprinter as _pp
except:
    pass
else:

    pp      = lambda *args, **kwargs: partial(_pp.pprint,  width=get_cols(), ribbon_width=get_cols(), indent=2)(*args, **kwargs)
    pformat = lambda *args, **kwargs: partial(_pp.pformat, width=get_cols(), ribbon_width=get_cols(), indent=2)(*args, **kwargs)
    exclude = []

    # HACK Workaround the builtin dataclasses extras not working correctly
    #   - Problem: dataclasses print with class name but an empty set of fields
    #   - Not the bug, but maybe usefully similar: https://github.com/tommikaikkonen/prettyprinter/pull/18
    exclude.append('dataclasses')
    try:
        import dataclasses
    except:
        pass
    else:
        import prettyprinter.extras.dataclasses
        _pp.register_pretty(predicate=_pp.extras.dataclasses.is_instance_of_dataclass)(
            lambda x, ctx: _pp.pretty_call(
                ctx,
                # type(x),  # Qualified name
                type(x).__name__,  # Unqualified name
                # **dataclasses.asdict(x),  # Don't do this -- it converts contained dataclasses into dicts
                **{f.name: getattr(x, f.name) for f in dataclasses.fields(x)},  # This preserves contained dataclasses
            )
        )

    _pp.install_extras(warn_on_error=False, exclude=exclude)
