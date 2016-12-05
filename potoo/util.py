import collections
import os
import pipes
import sys


_tonum   = lambda x: None if x is None else int(x)
_unlist  = lambda xs: None if len(xs) == 0 else xs[0]
get_rows = lambda: _tonum(_unlist(os.popen('stty size 2>/dev/null').read().split()[0:1]))
get_cols = lambda: _tonum(_unlist(os.popen('stty size 2>/dev/null').read().split()[1:2]))
# get_rows = lambda: int(os.popen('stty size 2>/dev/null').read().split()[0])  # XXX
# get_cols = lambda: int(os.popen('stty size 2>/dev/null').read().split()[1])  # XXX


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
