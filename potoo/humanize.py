import humanize


def naturalsize(size: float) -> str:
    """
    Ergonomics for humanize.naturalsize
    - Fix bug: negative sizes are always formatted as bytes
    - Abbrev 'Bytes' -> 'B'
    """
    s = (
        '-%s' % humanize.naturalsize(-size) if size < 0 else
        humanize.naturalsize(size)
    )
    s = s.replace(' Bytes', ' B')
    return s
