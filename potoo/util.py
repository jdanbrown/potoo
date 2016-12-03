import collections
import os
import pipes
import sys


get_rows = lambda: int(os.popen('stty size').read().split()[0])
get_cols = lambda: int(os.popen('stty size').read().split()[1])


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
