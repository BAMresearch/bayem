# third party imports
from scipy import stats


class LogPriorNormal:
    """Prior class for a normal distribution."""
    def __init__(self, prm_dict, ref_prm):
        """
        Parameters
        ----------
        prm_dict : dict
            Defines the prior parameters. In this case, the dictionary must
            have the form {'loc': <float>, 'scale': <float>}.
        ref_prm : string
            The name of the calibration-parameter the prior refers to.
        """
        self.loc = prm_dict['loc']
        self.scale = prm_dict['scale']
        self.ref_prm = ref_prm
        self.prior_type = "normal distribution"

    def __str__(self):
        """
        Allows printing an object of this class.

        Returns
        -------
        s : string
            A string containing the prior's attributes.
        """
        s = f"({self.prior_type} for '{self.ref_prm}', "
        s += f"loc={self.loc:.2f}, scale={self.scale:.2f})"
        return s

    def __call__(self, x):
        """
        Evaluates the log-PDF of the prior's normal distribution.

        Parameters
        ----------
        x : float
            A scalar value the prior should be evaluated at. This value refers
            to the prior's calibration-parameter ref_prm (see __init__). Note
            that the function returns to log-PDF.

        Returns
        -------
        float
            The logarithm of the prior's normal PDF evaluated at x.
        """
        return stats.norm.logpdf(x, self.loc, self.scale)


class LogPriorLognormal:
    """Prior class for a lognormal distribution."""
    def __init__(self, prm_dict, ref_prm):
        """
        Parameters
        ----------
        prm_dict : dict
            Defines the prior parameters. In this case, the dictionary must
            have the form {'loc': <float>, 'scale': <float>}.
        ref_prm : string
            The name of the calibration-parameter the prior refers to.
        """
        self.loc = prm_dict['loc']
        self.scale = prm_dict['scale']
        self.ref_prm = ref_prm
        self.prior_type = "lognormal distribution"

    def __str__(self):
        """
        Allows printing an object of this class.

        Returns
        -------
        s : string
            A string containing the prior's attributes.
        """
        s = f"({self.prior_type} for '{self.ref_prm}', "
        s += f"loc={self.loc:.2f}, scale={self.scale:.2f})"
        return s

    def __call__(self, x):
        """
        Evaluates the log-PDF of the prior's lognormal distribution.

        Parameters
        ----------
        x : float
            A scalar value the prior should be evaluated at. This value refers
            to the prior's calibration-parameter ref_prm (see __init__). Note
            that the function returns to log-PDF.

        Returns
        -------
        float
            The logarithm of the prior's lognormal-PDF evaluated at x.
        """
        return stats.lognorm.logpdf(x, self.loc, self.scale)


class LogPriorUniform:
    """Prior class for a uniform distribution."""

    def __init__(self, prm_dict, ref_prm):
        """
        Parameters
        ----------
        prm_dict : dict
            Defines the prior parameters. In this case, the dictionary must
            have the form {'loc': <float>, 'scale': <float>}.
        ref_prm : string
            The name of the calibration-parameter the prior refers to.
        """
        self.loc = prm_dict['loc']  # here, this refers to the lower bound
        self.scale = prm_dict['scale']  # here, this refers to the width
        self.ref_prm = ref_prm
        self.prior_type = "uniform distribution"

    def __str__(self):
        """
        Allows printing an object of this class.

        Returns
        -------
        s : string
            A string containing the prior's attributes.
        """
        # since it easier to interpret loc and scale are translated into the
        # lower and the upper bound of the prior's interval
        s = f"({self.prior_type} for '{self.ref_prm}', "
        s += f"lower={self.loc:.2f}, upper={self.loc + self.scale:.2f})"
        return s

    def __call__(self, x):
        """
        Evaluates the log-PDF of the prior's lognormal distribution.

        Parameters
        ----------
        x : float
            A scalar value the prior should be evaluated at. This value refers
            to the prior's calibration-parameter ref_prm (see __init__). Note
            that the function returns to log-PDF.

        Returns
        -------
        float
            The logarithm of the prior's lognormal-PDF evaluated at x.
        """
        return stats.uniform.logpdf(x, self.loc, self.scale)

class LogPriorWeibull:
    """Prior class for a three-parameter Weibull distribution."""

    def __init__(self, prm_dict, ref_prm):
        """
        Parameters
        ----------
        prm_dict : dict
            Defines the prior parameters. In this case, the dictionary must
            have the form {'loc': <float>, 'scale': <float>, 'shape': <float>}.
        ref_prm : string
            The name of the calibration-parameter the prior refers to.
        """
        self.loc = prm_dict['loc']
        self.scale = prm_dict['scale']
        self.shape = prm_dict['shape']
        self.ref_prm = ref_prm
        self.prior_type = "Weibull distribution"

    def __str__(self):
        """
        Allows printing an object of this class.

        Returns
        -------
        s : string
            A string containing the prior's attributes.
        """
        s = f"({self.prior_type} for '{self.ref_prm}', "
        s += f"scale={self.scale:.2f}, shape={self.shape:.2f}, "
        s += f"loc={self.loc:.2f})"
        return s

    def __call__(self, x):
        """
        Evaluates the log-PDF of the prior's lognormal distribution.

        Parameters
        ----------
        x : float
            A scalar value the prior should be evaluated at. This value refers
            to the prior's calibration-parameter ref_prm (see __init__). Note
            that the function returns to log-PDF.

        Returns
        -------
        float
            The logarithm of the prior's lognormal-PDF evaluated at x.
        """
        return stats.weibull_min.logpdf(x, self.shape, loc=self.loc,
                                        scale=self.scale)
