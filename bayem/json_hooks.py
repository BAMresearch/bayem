import json
import numpy as np

from .distributions import MVN, Gamma
from .vb import VBResult


class BayemEncoder(json.JSONEncoder):
    """
    Usage:
        string = json.dumps(obj, cls=bayem.vb.BayemEncoder, ...)
        with open(...) as f:
            json.dump(obj, f cls=bayem.vb.BayemEncoder, ...)

    Details:

    Out of the box, JSON can serialize
        dict, list, tuple, str, int, float, True/False, None

    To make our custom classes JSON serializable, we subclass from JSONEncoder
    and overwrite its `default` method to somehow represent our class with the
    types provided above.

    The idea is to serialize our custom classes (and numpy...) as a dict
    containing:
        key: unique string to represent the classe -- this helps us to indentify
             the class when _de_serializing in `bayem.vb.bayem_hook` below
        value: some json-serializable entries -- obj.__dict__ contains all
               members class members and is not optimal, but very convenient.
    https://stackoverflow.com/questions/3768895/how-to-make-a-class-json-serializable

    Note that the method is called recursively. To serialize MVN we call the
    `default` method below and see that json representation should contain
    the __dict__ of members. This dict also contains "mean" of type `np.array`.
    Thus, `default` is now called on `np.array` which is represented by a list
    of its values.

    """

    def default(self, obj):
        """
        `obj`:
            python object to serialize
        return:
            json (serializeable) reprensentation of `obj`
        """
        if isinstance(obj, VBResult):
            return {"vb.VBResult": obj.__dict__}

        if isinstance(obj, MVN):
            return {"vb.MVN": obj.__dict__}

        if isinstance(obj, Gamma):
            return {"vb.Gamma": obj.__dict__}

        if isinstance(obj, np.ndarray):
            return {"np.array": obj.tolist()}

        # `obj` is not one of our types? Fall back to superclass implementation.
        return json.JSONEncoder.default(self, obj)


def bayem_hook(dct):
    """
    `dct`:
        json reprensentation of an `obj` (a dict)
    `obj`:
        python object created from `dct`

    Usage:
        obj = json.loads(string, object_hook=bayem.vb.bayem_hook, ...)
        with open(...) as f:
            obj = json.load(f, object_hook=bayem.vb.bayem_hook, ...)

    Details:

    BayemEncoder stores all of our classes as dicts containing their members.
    This `object_hook` tries to convert those dicts back to actual python objects.

    Some fancy metaprogramming may help to avoid repetition. TODO?
    """
    if "vb.VBResult" in dct:
        result = VBResult()
        result.__dict__ = dct["vb.VBResult"]
        return result

    if "vb.MVN" in dct:
        mvn = MVN()
        mvn.__dict__ = dct["vb.MVN"]
        return mvn

    if "vb.Gamma" in dct:
        gamma = Gamma()
        gamma.__dict__ = dct["vb.Gamma"]
        return gamma

    if "np.array" in dct:
        return np.array(dct["np.array"])

    # Type not recognized, just return the dict.
    return dct
