import collections
from contextlib import contextmanager
from datetime import datetime
import os
import pipes
import shutil
import sys
import time
import traceback


def or_else(x, f):
    try:
        return f()
    except:
        return x


# 0 means use get_terminal_size(), ''/None means unlimited
get_rows = lambda: or_else(None, lambda: int(os.getenv('PD_ROWS'))) or 0  # noqa
get_cols = lambda: or_else(None, lambda: int(os.getenv('PD_COLS'))) or 0  # noqa

# XXX Back compat
stty_size = lambda: list(reversed(shutil.get_terminal_size()))  # noqa


def puts(x):
    print(x)
    return x


def singleton(cls):
    return cls()


def attrs(**kwargs):
    [keys, values] = list(zip(*kwargs.items())) or [[], []]
    return collections.namedtuple('attrs', keys)(*values)


def shell(cmd):
    print >>sys.stderr, 'shell: cmd[%s]' % cmd
    status = os.system(cmd)
    if status != 0:
        raise Exception('Exit status[%s] from cmd[%s]' % (status, cmd))


def mkdir_p(dir):
    os.system("mkdir -p %s" % pipes.quote(dir))  # Don't error like os.makedirs



# Do everything manually to avoid weird behaviors in curses impl below
def watch(period_s, f):
    try:
        os.system('stty -echo -cbreak')
        while True:
            nrows, ncols = stty_size()
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
