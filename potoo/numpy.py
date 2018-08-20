import numpy as np
from typing import Union

from potoo.util import get_cols

# Convenient shorthands for interactive use -- not recommended for durable code that needs to be read and maintained
A = np.array

def _float_format(width, precision):
    return lambda x: ('%%%s.%sg' % (width, precision)) % x


def set_display():
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.set_printoptions.html
    np.set_printoptions(
        linewidth=get_cols(),  # Default: 75
        precision=3,           # Default: 8; better magic than _float_format
        threshold=10000        # Default 1000; max total elements before summarizing cols and rows
        # formatter={'float_kind': _float_format(10, 3)}, # Default: magic in numpy.core.arrayprint
    )


def np_sample(
    x: np.ndarray,
    n: int = None,  # Like df.sample and np.random.choice
    frac: float = None,  # Like df.sample, unlike np.random.choice
    replace: bool = False,  # Like df.sample, unlike np.random.choice
    # weights: ...  # TODO
    random_state: Union[int, np.random.RandomState] = None,  # Like df.sample (requires more work with np.random)
) -> np.ndarray:
    """
    A variant of np.random.choice that presents an api more like pd.DataFrame.sample
    """

    # Args
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert (n is None) != (frac is None), f'Expected n[{n}] xor frac[{frac}]'
    if frac is not None:
        n = int(frac * len(x))
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    ix = random_state.choice(
        a=len(x),
        size=n,
        replace=replace,
        # p=...  # TODO
    )
    return x[ix]


# TODO Write tests
def np_sample_stratified(
    X: np.ndarray,
    y: np.ndarray,
    n: int = None,  # Like df.sample and np.random.choice
    frac: float = None,  # Like df.sample, unlike np.random.choice
    replace: bool = False,  # Like df.sample, unlike np.random.choice
    random_state: Union[int, np.random.RandomState] = None,  # Like df.sample (requires more work with np.random)
) -> (np.ndarray, np.ndarray):
    """
    Like np_sample, except:
    - Ensure approximate class balance
    - Ensure all classes have ≥1 sample (this is important to avoid various incidental complexity in e.g. sklearn utils)
    - Interpret n approximately as frac = n / len(X)
    """

    # Args
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    # To sample a total number n, sample frac = n / len(X)
    if n is not None:
        frac = n / len(X)
    if not (0 < frac <= 1):
        raise ValueError(f'Expected 0 < frac[{frac}] <= 1 (where n[{n}])')

    ixs = {}
    # For each class (y_)
    for y_ in y:
        # The X/y indexes for this class (y_)
        y_ix = (y == y_).nonzero()[0]
        # Sample them
        ixs[y_] = np_sample(
            y_ix,
            n=None,
            frac=frac,
            replace=replace,
            random_state=random_state,
        )
        # Ensure ≥1 sample per class
        if len(ixs[y_]) == 0:
            ixs[y_] = np_sample(
                y_ix,
                n=1,
                frac=None,
                replace=replace,
                random_state=random_state,
            )
    # Merge and permute the classes indexes
    ix = random_state.permutation([i for ix in ixs.values() for i in ix])

    return (X[ix], y[ix])
