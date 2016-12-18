import collections
import os
import pipes
import sys


def or_else(x, f):
    try:
        return f()
    except:
        return x


get_rows = lambda: or_else(100, lambda: int(os.popen('stty size 2>/dev/null').read().split()[0]))
get_cols = lambda: or_else(120, lambda: int(os.popen('stty size 2>/dev/null').read().split()[1]))


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
