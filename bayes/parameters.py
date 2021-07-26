from typing import Union, List
from numpy import typing as ndt

ParameterValue = Union[float, ndt.ArrayLike]


class ParameterList:
    """
    The ParameterList serves as an input to the user-defined models. It is
    basically a name:value-dict that allows the user to access the parameters
    by name instead of some vector index, which could read:

        def my_model(prm):
            return prm["slope"] * some_coordinates + prm["offset"]

    """

    def __init__(self) -> None:
        self.p = {}

    def define(self, name: str, value: ParameterValue = None) -> None:
        self.p[name] = value

    def __getitem__(self, name: str) -> ParameterValue:
        return self.p[name]

    def __contains__(self, name: str) -> bool:
        return name in self.p

    def __setitem__(self, name: str, value: ParameterValue) -> None:
        """
        Calling parameter_list["A"]=0. when there is no parameter "A" defined
        may hide some bugs in the user code. Thus, we force parameters to be
        defined via `self.define(name)` before accessing it.
        """
        if name not in self:
            raise Exception("Call .define to define new parameters.")
        self.define(name, value)

    def __add__(self, other: "ParameterList") -> "ParameterList":
        """
        Adding two ParameterLists can be convenient of nested models. An
        example could be a model error that combines a forward_model and
        a sensor_data_model like:

            class MyModelError:
                def __init__(self, forward_model, sensor_data_model):
                    self.fw = forward_model
                    self.data = sensor_data_model
                    self.parameter_list = self.fw.parameter_list + self.data.parameter_list
                                                                 ^
                                    this "+" is defined here ----|
        """
        concat = self._copy()
        for name, value in other.p.items():
            concat.define(name, value)
        return concat

    def _copy(self):
        new = ParameterList()
        for name, value in self.p.items():
            new.define(name, value)
        return new

    def overwrite_with(self, other: "ParameterList") -> "ParameterList":
        """ """
        concat = self._copy()
        for name, value in other.p.items():
            assert name in concat
            concat.define(name, value)
        return concat

    def with_value(self, name: str, value: ParameterValue) -> "ParameterList":
        """ """
        new = self._copy()
        new[name] = value
        return new

    def __str__(self) -> str:
        s = ""
        for name, value in self.p.items():
            s += f"{name:20s} {value}\n"
        return s

    @property
    def names(self) -> List[str]:
        return list(self.p.keys())
