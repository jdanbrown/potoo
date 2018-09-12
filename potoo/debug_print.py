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


def _debug_print(*args, _lines=False, **kwargs):
    caller = inspect.stack(context=0)[2]
    msg_vals = [*args, *['%s=%s' % (k, v) for k, v in kwargs.items()]]
    msg = (
        '' if not msg_vals else
        ': %s' % ', '.join(map(str, msg_vals)) if not _lines else
        '\n  %s' % '\n  '.join(map(str, msg_vals))
    )
    print('PRINT [%s] [%s:%s] %s%s' % (
        datetime.utcnow().isoformat()[11:23],  # Strip date + micros
        os.path.basename(caller.filename),
        caller.lineno,
        caller.function,
        msg,
    ))


class module(types.ModuleType):

    def __init__(self):
        super().__init__(__name__)

    def __call__(self, *args, **kwargs):
        _debug_print(*args, **kwargs)


sys.modules[__name__] = module()
