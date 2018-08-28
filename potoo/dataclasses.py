from collections import abc, OrderedDict
import json
import sys
from typing import Iterable

from attrdict import AttrDict
import dataclasses
from potoo.util import get_cols


class DataclassUtil:
    """Things I wish all dataclasses had"""

    @classmethod
    def field_names(cls, **filters) -> Iterable[str]:
        return [
            x.name
            for x in dataclasses.fields(cls)
            if all(getattr(x, k) == v for k, v in filters.items())
        ]

    def replace(self, **kwargs) -> 'Self':
        return dataclasses.replace(self, **kwargs)

    def asdict(self) -> dict:
        """Convert to dict preserving field order, e.g. for df rows"""
        return OrderedDict(dataclasses.asdict(self))

    def asattr(self) -> AttrDict:
        return AttrDict(self.asdict())

    def __sizeof__(self):
        try:
            from dask.sizeof import sizeof
        except:
            sizeof = sys.getsizeof
        return sizeof(list(self.asdict().items()))


class DataclassAsDict(abc.MutableMapping):
    """Expose a dict interface for the fields of a dataclass"""

    def __getitem__(self, k):
        return self.__dict__.__getitem__(k)

    def __setitem__(self, k, v):
        return self.__dict__.__setitem__(k, v)

    def __delitem__(self, k):
        return self.__dict__.__delitem__(k)

    def __iter__(self):
        return self.__dict__.__iter__()

    def __len__	(self):
        return self.__dict__.__len__()
