class ParameterList:
    """
    The ParameterList serves as an input to the user-defined models. It is 
    basically a name:value-dict that allows the user to access the parameters
    by name instead of some vector index, which could read:

        def my_model(prm):
            return prm["slope"] * some_coordinates + prm["offset"]

    The ParameterList differs from a dictionary in having additional features
    like allowing to add two ParameterLists (see __add__) or the way how new
    entries are defined (see define).
    """

    def __init__(self):
        """The name:value pairs are stored as a dict in the attribute p."""
        self.p = {}

    def define(self, name, value=None):
        """This method is intended to add a new parameter to the object."""
        self.p[name] = value

    def __getitem__(self, name):
        """Access the value of a parameter by an expression like prm["A"]."""
        return self.p[name]

    def __setitem__(self, name, value):
        """
        Calling prm["A"]=0 when there is no parameter "A" defined yet may hide
        some bugs in the user code. Thus, by defining this method, we force new
        parameters to be set via the define-method (prm.define("A")=0).
        """
        if name not in self:
            raise Exception("Call .define to define new parameters.")
        self.define(name, value)

    def __contains__(self, name):
        """Check if parameter "A" is already defined in prm by "A" in prm."""
        return name in self.p

    def __add__(self, other):
        """
        Adding two ParameterLists can be convenient of nested models. An 
        example could be a model error that combines a forward_model and
        a sensor_data_model like:
            
        class MyModelError:
            def __init__(self, forward_model, sensor_data_model):
                self.fw = forward_model
                self.data = sensor_data_model
                self.parameter_list = self.fw.prm + self.data.prm
                                                 /\
                     this "+" is defined here ---|
        """
        concat = ParameterList()
        for name, value in self.p.items():
            concat.define(name, value)
        for name, value in other.p.items():
            concat.define(name, value)
        return concat
    
    def __str__(self):
        """Defines the printed output when calling print(prm)."""
        s = ""
        for name, value in self.p.items():
            s += f"{name:20s} {value}\n"
        return s

    @property
    def names(self):
        """Calling prm.names and returns a list with all parameter names."""
        return list(self.p.keys())
