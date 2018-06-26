import collections
from contextlib import contextmanager
from datetime import datetime
from functools import partial
import numbers
import os
import pipes
import shlex
import shutil
import subprocess
import sys
import time
import traceback


# In pandas, 0 means use get_terminal_size(), ''/None means unlimited
get_term_size = lambda: shutil.get_terminal_size()  # ($COLUMNS else detect dynamically, $LINES else detect dynamically)  # noqa
get_rows = lambda: get_term_size().lines            # $LINES else detect dynamically                                      # noqa
get_cols = lambda: get_term_size().columns          # $COLUMNS else detect dynamically                                    # noqa


def puts(x):
    print(x)
    return x


def dirs(x, _=False, __=False):
    return {k: getattr(x, k) for k in dir(x) if all([
        not k.startswith('__') or __,
        not k.startswith('_') or _,
    ])}
# "Tests"
# print(); pp(list(dirs(x, _=False).keys()))
# print(); pp(list(dirs(x, __=False).keys()))
# print(); pp(list(dirs(x).keys()))


def singleton(cls):
    return cls()


def strip_startswith(s: str, startswith: str, check=False) -> str:
    if s.startswith(startswith):
        return s[len(startswith):]
    else:
        if check:
            raise ValueError(f"s[{s!r}] doesn't start with startswith[{startswith!r}]")
        return s


def strip_endswith(s: str, endswith: str, check=False) -> str:
    if s.endswith(endswith):
        return s[:len(endswith)]
    else:
        if check:
            raise ValueError(f"s[{s!r}] doesn't end with endswith[{endswith!r}]")
        return s


def or_else(x, f):
    try:
        return f()
    except:
        return x


def raise_(e):
    """Raise in an expression instead of a statement"""
    raise e


# XXX Use AttrDict
# def attrs(**kwargs):
#     [keys, values] = list(zip(*kwargs.items())) or [[], []]
#     return collections.namedtuple('attrs', keys)(*values)


class AttrContext:
    """Expose an object's attrs as a context manager. Useful for global singleton config objects."""

    @contextmanager
    def context(self, **attrs):
        self._raise_on_unknown_attrs(attrs)
        to_restore = {k: v for k, v in self.__dict__.items() if k in attrs}  # Don't overwrite unrelated mutations
        self.__dict__.update(attrs)
        try:
            yield
        finally:
            self.__dict__.update(to_restore)

    def __call__(self, **attrs):
        """Like .set when called, like .context when used in a `with`"""
        self._raise_on_unknown_attrs(attrs)
        to_restore = {k: v for k, v in self.__dict__.items() if k in attrs}  # Don't overwrite unrelated mutations
        self.__dict__.update(attrs)
        @contextmanager
        def ctx():
            try:
                yield
            finally:
                self.__dict__.update(to_restore)
        return ctx()

    def set(self, **attrs):
        self._raise_on_unknown_attrs(attrs)
        self.__dict__.update(attrs)

    def _raise_on_unknown_attrs(self, attrs):
        unknown = {k: v for k, v in attrs.items() if k not in self.__dict__}
        if unknown:
            raise ValueError(f'Unknown attrs: {unknown}')


def shell(cmd, **kwargs):
    cmd = cmd % {
        k: shlex.quote(str(v))
        for k, v in kwargs.items()
    }
    print(f'$ {cmd}', file=sys.stderr)
    return subprocess.run(cmd, shell=True, check=True)


def mkdir_p(dir):
    os.system("mkdir -p %s" % pipes.quote(dir))  # Don't error like os.makedirs


def timed_print(f, **kwargs):
    elapsed, x = timed_format(f, **kwargs)
    print(elapsed)
    return x


def timed_format(f, **kwargs):
    elapsed_s, x = timed(f, **kwargs)
    elapsed = '[%s]' % format_duration(elapsed_s)
    return elapsed, x


def timed(f, *args, finally_=lambda elapsed_s: None, **kwargs):
    start_s = time.time()
    try:
        x = f(*args, **kwargs)
    finally:
        elapsed_s = time.time() - start_s
        finally_(elapsed_s)
    return elapsed_s, x


def format_duration(secs):
    """
    >>> format_duration(0)
    '00:00'
    >>> format_duration(1)
    '00:01'
    >>> format_duration(100)
    '01:40'
    >>> format_duration(10000)
    '02:46:40'
    >>> format_duration(1000000)
    '277:46:40'
    >>> format_duration(0.0)
    '00:00.000'
    >>> format_duration(0.5)
    '00:00.500'
    >>> format_duration(12345.6789)
    '03:25:45.679'
    >>> format_duration(-1)
    '-00:01'
    >>> format_duration(-10000)
    '-02:46:40'
    """
    if secs < 0:
        return '-' + format_duration(-secs)
    else:
        s = int(secs) % 60
        m = int(secs) // 60 % 60
        h = int(secs) // 60 // 60
        res = ':'.join('%02.0f' % x for x in (
            [m, s] if h == 0 else [h, m, s]
        ))
        if isinstance(secs, float):
            ms = round(secs % 1, 3)
            res += ('%.3f' % ms)[1:]
        return res


def round_sig(x, n):
    if isinstance(x, numbers.Rational):
        # Don't touch ints: this is technically incorrect ("round significant digits") but the less surprising behavior
        # if you expect "round" always means "round floats"
        return x
    elif isinstance(x, numbers.Real):
        return type(x)(f'%.{n}g' % x)
    else:
        # Use complex(...) instead of type(x)(...), since complex parses from str but e.g. np.complex128 doesn't.
        # Use x.__format___ instead of '%.3g' % ..., since the latter doesn't support complex numbers (dunno why).
        return complex(x.__format__(f'.{n}g'))


def deep_round_sig(x, n):
    recur = partial(deep_round_sig, n=n)
    base = partial(round_sig, n=n)
    if isinstance(x, dict):
        return type(x)({k: recur(v) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)(map(recur, x))
    elif _is_np_ndarray(x):
        # Can't `type(x)(...)` to construct, and have to `list(...)` the iter
        import numpy as np
        return np.array(list(map(recur, x)))
    elif _is_pd_series(x):
        return x.map(recur)
    elif _is_pd_dataframe(x):
        return x.applymap(recur)
    # TODO Add more...
    elif isinstance(x, numbers.Number):
        return base(x)
    else:
        return x


def _is_np_ndarray(x):
    try:
        import numpy as np
        return isinstance(x, np.ndarray)
    except ModuleNotFoundError:
        return False


def _is_pd_series(x):
    try:
        import pandas as pd
        return isinstance(x, pd.Series)
    except ModuleNotFoundError:
        return False


def _is_pd_dataframe(x):
    try:
        import pandas as pd
        return isinstance(x, pd.DataFrame)
    except ModuleNotFoundError:
        return False


#
# watch (requires curses)
#


# Do everything manually to avoid weird behaviors in curses impl below
def watch(period_s, f):
    try:
        os.system('stty -echo -cbreak')
        while True:
            ncols, nrows = get_term_size()
            try:
                s = str(f())
            except Exception:
                s = traceback.format_exc()
            if not s.endswith('\n'):
                s += '\n'  # Ensure cursor is on next blank line
            lines = s.split('\n')
            lines = [
                datetime.now().isoformat()[:-3],  # Drop micros
                '',
            ] + lines
            os.system('tput cup 0 0')
            for row_i in range(nrows):
                if row_i < len(lines):
                    line = lines[row_i][:ncols]
                else:
                    line = ''
                trailing_space = ' ' * max(0, ncols - len(line))
                print(
                    line + trailing_space,
                    end='\n' if row_i < nrows - 1 else '',
                    flush=True,
                )
            os.system(f'tput cup {nrows - 1} {ncols - 1}')
            time.sleep(period_s)
    except KeyboardInterrupt:
        pass
    finally:
        os.system('stty sane 2>/dev/null')
        print()


# # FIXME stdscr.cbreak() barfs from within ipython, and omitting it soemtimes drops leading spaces
# def watch(period_s, f):
#     with use_curses() as (curses, stdscr):
#         try:
#             while True:
#                 curses.noecho()  # Don't echo key presses
#                 curses.cbreak()  # Don't buffer input until enter [also avoid addstr dropping leading spaces]
#                 max_y, max_x = stdscr.getmaxyx()
#                 stdscr.clear()
#                 stdscr.addnstr(0, 0, datetime.now().isoformat()[:-3], max_x)  # Drop micros
#                 try:
#                     s = str(f())
#                 except Exception:
#                     s = traceback.format_exc()
#                 if not s.endswith('\n'):
#                     s += '\n'  # Ensure cursor is on next blank line
#                 y = 2
#                 for line in s.split('\n'):
#                     # Don't addstr beyond max_y
#                     if y <= max_y - 2:
#                         # Don't addstr beyond max_x
#                         line = line[:max_x]
#                         try:
#                             # All chars must be within (max_y, max_x), else you'll get unhelpful "returned ERR" errors
#                             stdscr.addstr(y, 0, line)
#                         except:
#                             # Raise helpful error in case we use addstr wrong (it's very finicky with y/x)
#                             raise Exception('Failed to addstr(%r)' % line)
#                         y += 1
#                 stdscr.refresh()
#                 time.sleep(period_s)
#         except KeyboardInterrupt:
#             pass


@singleton
class use_curses:

    def __init__(self):
        self._stdscr = None

    @contextmanager
    def __call__(self):
        # Don't import until used, in case curses does weird things on some platform or in some environment
        import curses
        if self._stdscr is None:
            # Warning: if you call initscr twice you'll break curses for the rest of the current process
            #   - e.g. breaks under ipython %autoreload
            self._stdscr = curses.initscr()
        try:
            yield (curses, self._stdscr)
        finally:
            # Warning: if you call endwin you'll never be able to use curses again for the rest of the current process
            # curses.endwin()
            # Do `stty sane` instead
            os.system('stty sane 2>/dev/null')
            # Reset cursor to x=0
            print('')
