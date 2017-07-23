import collections
import os
import pipes
import sys


def or_else(x, f):
    try:
        return f()
    except:
        return x


stty_size = lambda: [int(x) for x in os.popen('stty size 2>/dev/null').read().split()]
get_rows = lambda: or_else(None, lambda: int(os.getenv('PD_ROWS'))) or or_else(None, lambda: stty_size()[0]) or 100
get_cols = lambda: or_else(None, lambda: int(os.getenv('PD_COLS'))) or or_else(None, lambda: stty_size()[1]) or 120


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
