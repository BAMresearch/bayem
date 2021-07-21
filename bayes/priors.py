# third party imports
from scipy import stats


class LogPriorTemplate:
    """
    Template class for prior definitions. The required return value of its
    __call__ function is the logarithm of the evaluated distribution function,
    hence the name LogPriorTemplate. Note that the main motivation of how this
    class is defined is to avoid storing any numeric values for any of the
    priors parameters.
    """
    def __init__(self, ref_prm, prms_def, name, prior_type):
        """
        Parameters
        ----------
        ref_prm : string
            The name of the calibration-parameter the prior refers to.
        prms_def : list
            A list of strings defining the priors parameters.
        name : string
            Defining the priors name.
        prior_type : string
            Stating the prior's type, e.g. "normal distribution"
        """
        # write arguments to attributes
        self.ref_prm = ref_prm
        self.prms = prms_def
        self.name = name
        self.prior_type = prior_type  # "normal distribution"

        # this attribute defines the parameters given to the __call__-method
        self.prms_def = [ref_prm] + prms_def

    def __str__(self):
        """
        Allows printing an object of this class.

        Returns
        -------
        s : string
            A string containing the prior's attributes.
        """
        s = f"{self.prior_type} for '{self.ref_prm}', prms={self.prms_def}"
        return s


class LogPriorNormal(LogPriorTemplate):
    """Prior class for a normal distribution."""
    def __init__(self, ref_prm, prms_def, name):
        """Check out the explanations for LogPriorTemplate given above"""
        super().__init__(ref_prm, prms_def, name, "normal distribution")

    def __call__(self, prms, method):
        """
        Evaluates the log-PDF of the prior's normal distribution.

        Parameters
        ----------
        prms : ParameterList-object
            Dictionary-like object containing parameter name:value pairs
        method : string
            Method from the used scipy distribution (e.g. 'pdf', 'logpdf', etc.)

        Returns
        -------
        float
            The logarithm of the prior's normal PDF evaluated at x.
        """
        x = prms[self.ref_prm]
        loc = prms[f"loc_{self.ref_prm}"]
        scale = prms[f"scale_{self.ref_prm}"]
        return eval(f"stats.norm.{method}")(x, loc, scale)


class LogPriorLognormal(LogPriorTemplate):
    """Prior class for a lognormal distribution."""
    def __init__(self, ref_prm, prms_def, name):
        """Check out the explanations for LogPriorTemplate given above"""
        super().__init__(ref_prm, prms_def, name, "lognormal distribution")

    def __call__(self, prms, method):
        """
        Evaluates the log-PDF of the prior's lognormal distribution.

        Parameters
        ----------
        prms : ParameterList-object
            Dictionary-like object containing parameter name:value pairs
        method : string
            Method from the used scipy distribution (e.g. 'pdf', 'logpdf', etc.)

        Returns
        -------
        float
            The logarithm of the prior's normal PDF evaluated at x.
        """
        x = prms[self.ref_prm]
        loc = prms[f"loc_{self.ref_prm}"]
        scale = prms[f"scale_{self.ref_prm}"]
        return eval(f"stats.lognorm.{method}")(x, loc, scale)


class LogPriorUniform(LogPriorTemplate):
    """Prior class for a uniform distribution."""
    def __init__(self, ref_prm, prms_def, name):
        """Check out the explanations for LogPriorTemplate given above"""
        super().__init__(ref_prm, prms_def, name, "uniform distribution")

    def __call__(self, prms, method):
        """
        Evaluates the log-PDF of the prior's uniform distribution.

        Parameters
        ----------
        prms : ParameterList-object
            Dictionary-like object containing parameter name:value pairs
        method : string
            Method from the used scipy distribution (e.g. 'pdf', 'logpdf', etc.)

        Returns
        -------
        float
            The logarithm of the prior's normal PDF evaluated at x.
        """
        x = prms[self.ref_prm]
        low = prms[f"low_{self.ref_prm}"]
        high = prms[f"high_{self.ref_prm}"]
        return eval(f"stats.uniform.{method}")(x, loc=low, scale=high-low)

class LogPriorWeibull(LogPriorTemplate):
    """Prior class for a three-parameter Weibull distribution."""
    def __init__(self, ref_prm, prms_def, name):
        """Check out the explanations for LogPriorTemplate given above"""
        super().__init__(ref_prm, prms_def, name, "uniform distribution")

    def __call__(self, prms, method):
        """
        Evaluates the log-PDF of the prior's Weibull distribution.

        Parameters
        ----------
        prms : ParameterList-object
            Dictionary-like object containing parameter name:value pairs
        method : string
            Method from the used scipy distribution (e.g. 'pdf', 'logpdf', etc.)

        Returns
        -------
        float
            The logarithm of the prior's normal PDF evaluated at x.
        """
        x = prms[self.ref_prm]
        shape = prms[f"shape_{self.ref_prm}"]
        scale = prms[f"scale_{self.ref_prm}"]
        loc = prms[f"loc_{self.ref_prm}"]
        return eval(f"stats.weibull_min.{method}")(x, shape, scale=scale,
                                                   loc=loc)

