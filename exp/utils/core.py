import re
from typing import Iterable


def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]


def camel2snake(name):
    """
    Helper function that simply turns such string names like "AbcDefGh" into "abc_def_gh"
    """
    _camel_re1 = re.compile('(.)([A-Z][a-z]+)')
    _camel_re2 = re.compile('([a-z0-9])([A-Z])')
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()


class CancelTrainException(Exception): 
    pass


class CancelEpochException(Exception): 
    pass


class CancelBatchException(Exception): 
    pass
