from contextlib import contextmanager


class dynvar:

    def __init__(self, value):
        self._values = [value]

    @contextmanager
    def __call__(self, value):
        self._values.append(value)
        try:
            yield
        finally:
            self._values.pop(-1)

    def value(self):
        return self._values[-1]
