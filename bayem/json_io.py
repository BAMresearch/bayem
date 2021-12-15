import json
from dataclasses import asdict

import numpy as np

from .distributions import MVN, Gamma
from .vba import VBResult, VBOptions


class BayemEncoder(json.JSONEncoder):
    """
    Usage:
        string = json.dumps(obj, cls=bayem.BayemEncoder, ...)
        with open(...) as f:
            json.dump(obj, f cls=bayem.BayemEncoder, ...)

    Details:

    Out of the box, JSON can serialize
        dict, list, tuple, str, int, float, True/False, None

    To make our custom classes JSON serializable, we subclass from JSONEncoder
    and overwrite its `default` method to somehow represent our class with the
    types provided above.

    The idea is to serialize our custom classes (and numpy...) as a dict
    containing:
        key: unique string to represent the classe -- this helps us to indentify
             the class when _de_serializing in `bayem.bayem_hook` below
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
            return {"bayem.VBResult": obj.__dict__}

        if isinstance(obj, VBOptions):
            return {"bayem.VBOptions": asdict(obj)}

        if isinstance(obj, MVN):
            return {"bayem.MVN": obj.__dict__}

        if isinstance(obj, Gamma):
            return {"bayem.Gamma": obj.__dict__}

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
        obj = json.loads(string, object_hook=bayem.bayem_hook, ...)
        with open(...) as f:
            obj = json.load(f, object_hook=bayem.bayem_hook, ...)

    Details:

    BayemEncoder stores all of our classes as dicts containing their members.
    This `object_hook` tries to convert those dicts back to actual python objects.

    Some fancy metaprogramming may help to avoid repetition. TODO?
    """
    if "bayem.VBResult" in dct:
        result = VBResult(None)
        result.__dict__ = dct["bayem.VBResult"]
        return result

    if "bayem.VBOptions" in dct:
        return VBOptions(**dct["bayem.VBOptions"])

    if "bayem.MVN" in dct:
        mvn = MVN()
        mvn.__dict__ = dct["bayem.MVN"]
        return mvn

    if "bayem.Gamma" in dct:
        gamma = Gamma()
        gamma.__dict__ = dct["bayem.Gamma"]
        return gamma

    if "np.array" in dct:
        return np.array(dct["np.array"])

    # Type not recognized, just return the dict.
    return dct


def save_json(obj, filename=None):
    """
    Saves an `obj` (possibly containing VB classes) to `filename` via json or
    returns the json string.
    """
    s = json.dumps(obj, cls=BayemEncoder, indent=2)
    if filename is not None:
        with open(filename, "w") as f:
            f.write(s)
    return s


def load_json(filename_or_string):
    """
    Loads an object (possibly containing VB classes) from `filename_or_string`
    via json.
    """
    if str(filename_or_string).endswith(".json"):
        with open(filename_or_string, "r") as f:
            string = f.read()
    else:
        string = filename_or_string

    return json.loads(string, object_hook=bayem_hook)
