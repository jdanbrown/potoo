import numpy as np

from jdanbrown.util import get_cols


def _float_format(width, precision):
    return lambda x: ('%%%s.%sg' % (width, precision)) % x


def set_display():
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.set_printoptions.html
    np.set_printoptions(
        linewidth = get_cols(),  # Default: 75
        precision = 3,           # Default: 8; better magic than _float_format
        # formatter = {'float_kind': _float_format(10, 3)}, # Default: magic in numpy.core.arrayprint
    )
