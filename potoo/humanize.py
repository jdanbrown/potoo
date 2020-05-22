import humanize

from potoo.util import str_strip_endswith


def naturalsize_binary(size: float, **kwargs) -> str:
    """
    Ergonomics for humanize.naturalsize
    - Default to binary=True
    """
    return naturalsize(size, **{
        'binary': True,
        **kwargs,
    })


def naturalsize(size: float, **kwargs) -> str:
    """
    Ergonomics for humanize.naturalsize
    - Fix bug: negative sizes are always formatted as bytes
    - Abbrev 'Bytes' -> 'B'
    """
    s = (
        '-%s' % humanize.naturalsize(-size, **kwargs) if size < 0 else
        humanize.naturalsize(size, **kwargs)
    )
    s = s.replace(' Bytes', ' B')
    s = s.replace(' Byte', ' B')
    return s


def naturalsize_no_suffix(size: float, **kwargs) -> str:
    s = naturalsize(size)
    s = str_strip_endswith(s, 'B', 'Bytes', 'Byte').rstrip()
    return s
