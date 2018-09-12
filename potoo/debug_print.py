"""
printf debugging that includes the source filename, line number, and function name

Example usage:
    from potoo import debug_print
    debug_print()                                  # Print enough info to know what line you hit
    debug_print('x exists')                        # Add a msg
    debug_print('x exists', x)                     # Add a msg with data
    debug_print(x=x, i=i, **kwargs)                # Just dump lots of stuff
    debug_print(x=x, i=i, **kwargs, _lines=False)  # Give each value its own line
"""

from datetime import datetime
import inspect
import os
import sys
import types


def _debug_print(*args, _lines=False, _depth=1, **kwargs):
    caller = inspect.stack(context=0)[_depth]
    msg_vals = [*args, *['%s=%r' % (k, v) for k, v in kwargs.items()]]
    msg = (
        '' if not msg_vals else
        ': %s' % ', '.join(map(str, msg_vals)) if not _lines else
        '\n  %s' % '\n  '.join(map(str, msg_vals))
    )
    print('%s [%s] [%s:%s] %s%s' % (
        '%-8s' % 'PRINT',  # %-8s like a typical logging format (which has to fit 'CRITICAL')
        datetime.utcnow().isoformat()[11:23],  # Strip date + micros
        os.path.basename(caller.filename),
        caller.lineno,
        caller.function,
        msg,
    ))
    # Return first arg (like puts)
    #   - Support args (e.g. debug_print(x)) as well as kwargs (e.g. debug_print(x=x))
    if args:
        return args[0]
    elif kwargs:
        return list(kwargs.values())[0]
    else:
        return None


class module(types.ModuleType):

    def __init__(self):
        super().__init__(__name__)

    def __call__(self, *args, _depth=2, **kwargs):
        return _debug_print(*args, **kwargs, _depth=_depth)

sys.modules[__name__] = module()
