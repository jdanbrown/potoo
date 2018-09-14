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


def _debug_print(*args, _lines=False, _depth=1, _quiet=False, _utc=False, **kwargs):

    if not _quiet:

        # Inspect stack for caller's frame
        caller = inspect.stack(context=0)[_depth]

        # Format and print
        level = '%-8s' % 'PRINT'  # %-8s like a typical logging format's log level (which has to fit 'CRITICAL')
        timestamp = (datetime.utcnow() if _utc else datetime.now()).isoformat()[11:23]  # Strip date + micros
        pid = os.getpid()
        lineno = caller.lineno
        module = inspect.getmodule(caller.frame)
        module = module and module.__name__
        function = caller.function
        module_function = f'{module}/{function}' if module else function
        msg_vals = [*args, *['%s=%r' % (k, v) for k, v in kwargs.items()]]
        msg = (
            '' if not msg_vals else
            ': %s' % ', '.join(map(str, msg_vals)) if not _lines else
            '\n  %s' % '\n  '.join(map(str, msg_vals))
        )
        print(f'{level} [{timestamp}] [{pid:5d}]{lineno:4d} {module_function}{msg}')

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
        """
        Avoid one step in the import, e.g.
            from potoo import debug_print
            debug_print('foo', x=x, y=y)
        """
        return _debug_print(*args, **kwargs, _depth=_depth)

    def quiet(self, *args, _depth=2, **kwargs):
        """
        Temporarily disable a debug_print in expression context, e.g.
            x + debug_print.quiet(y=y)
        """
        return _debug_print(*args, **kwargs, _depth=_depth, _quiet=True)

sys.modules[__name__] = module()
