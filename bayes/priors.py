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
            A list of strings defining the priors parameters. Note that the
            order is important! Check out the explanation of the attribute
            self.prms_def given below.
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

        # this attribute defines how the parameter vector given to the __call__
        # method is going to be set up; if self.prms_def = ["a", "mu_a", "sd_a"]
        # then the prms argument of __call__ will be a numeric vector giving a
        # value for "a" at the first position, a value for "mu_a" at the second
        # position and a value for "sd_a" at the third position; make sure the
        # call method processes the prms argument correspondingly!
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

    def __call__(self, prms):
        """
        Evaluates the log-PDF of the prior's normal distribution.

        Parameters
        ----------
        prms : array_like
            A numeric vector to be interpreted according to self.prms. The first
            element refers to the priors reference parameter (e.g. if the prior
            refers to some parameter "b" of the inference problem, then the
            first element of prms is a value for b). The following elements are
            the priors parameter.

        Returns
        -------
        float
            The logarithm of the prior's normal PDF evaluated at x.
        """
        x = prms[0]
        loc = prms[1]
        scale = prms[2]
        return stats.norm.logpdf(x, loc, scale)


class LogPriorLognormal(LogPriorTemplate):
    """Prior class for a lognormal distribution."""
    def __init__(self, ref_prm, prms_def, name):
        """Check out the explanations for LogPriorTemplate given above"""
        super().__init__(ref_prm, prms_def, name, "lognormal distribution")

    def __call__(self, prms):
        """
        Evaluates the log-PDF of the prior's lognormal distribution.

        Parameters
        ----------
        prms : array_like
            A numeric vector to be interpreted according to self.prms. The first
            element refers to the priors reference parameter (e.g. if the prior
            refers to some parameter "b" of the inference problem, then the
            first element of prms is a value for b). The following elements are
            the priors parameter.

        Returns
        -------
        float
            The logarithm of the prior's normal PDF evaluated at x.
        """
        x = prms[0]
        loc = prms[1]
        scale = prms[2]
        return stats.lognorm.logpdf(x, loc, scale)


class LogPriorUniform(LogPriorTemplate):
    """Prior class for a uniform distribution."""
    def __init__(self, ref_prm, prms_def, name):
        """Check out the explanations for LogPriorTemplate given above"""
        super().__init__(ref_prm, prms_def, name, "uniform distribution")

    def __call__(self, prms):
        """
        Evaluates the log-PDF of the prior's uniform distribution.

        Parameters
        ----------
        prms : array_like
            A numeric vector to be interpreted according to self.prms. The first
            element refers to the priors reference parameter (e.g. if the prior
            refers to some parameter "b" of the inference problem, then the
            first element of prms is a value for b). The following elements are
            the priors parameter.

        Returns
        -------
        float
            The logarithm of the prior's normal PDF evaluated at x.
        """
        x = prms[0]
        loc = prms[1]
        scale = prms[2]
        return stats.uniform.logpdf(x, loc, scale)

class LogPriorWeibull(LogPriorTemplate):
    """Prior class for a three-parameter Weibull distribution."""
    def __init__(self, ref_prm, prms_def, name):
        """Check out the explanations for LogPriorTemplate given above"""
        super().__init__(ref_prm, prms_def, name, "uniform distribution")

    def __call__(self, prms):
        """
        Evaluates the log-PDF of the prior's Weibull distribution.

        Parameters
        ----------
        prms : array_like
            A numeric vector to be interpreted according to self.prms. The first
            element refers to the priors reference parameter (e.g. if the prior
            refers to some parameter "b" of the inference problem, then the
            first element of prms is a value for b). The following elements are
            the priors parameter.

        Returns
        -------
        float
            The logarithm of the prior's normal PDF evaluated at x.
        """
        x = prms[0]
        shape = prms[1]
        scale = prms[2]
        loc = prms[3]
        return stats.weibull_min.logpdf(x, shape, scale=scale, loc=loc)
