import numpy as np
from scipy.optimize import brenth
from scipy.stats import gamma, multivariate_normal, norm
from tabulate import tabulate


class MVN:
    def __init__(self, mean=[0.0], precision=[[1.0]], name="MVN", parameter_names=None):
        """
        Creates a N-dimensional multivariate normal distribution from provided
        `mean` μ and `precision` L with the PDF

           f(x) = (2π)^(-N/2) det(L)^(1/2) exp[-1/2 (x - μ)^T L (x - μ)]

        mean:
            vector of N means μ

        precision:
            precision matrix L with dimension NxN

        name:
            name of the distribution, e.g. to indicate "prior" or "posterior"

        parameter_names:
            vector of N strings that name the N parameters for convenient
            access via `self.named_mean`
        """
        self.mean = np.atleast_1d(mean).astype(float)
        self.precision = np.atleast_2d(precision).astype(float)
        self.name = name
        if parameter_names is None:
            self.parameter_names = [
                r"$\theta_{" + str(i) + "}$" for i in range(len(self.mean))
            ]
        else:
            self.parameter_names = parameter_names

        assert len(self.mean) == len(self.precision)

        if self.parameter_names is not None:
            assert len(self.parameter_names) == len(self.mean)

    @classmethod
    def FromMeanStd(cls, mean, stds, **kwargs):
        prec = np.diag([1 / sd ** 2 for sd in stds])
        return cls(mean, prec, **kwargs)

    def __len__(self):
        return len(self.mean)

    def index(self, parameter_name):
        return self.parameter_names.index(parameter_name)

    def named_mean(self, parameter_name):
        return self.mean[self.index(parameter_name)]

    def named_sd(self, parameter_name):
        return self.std_diag[self.index(parameter_name)]

    @property
    def std_diag(self):
        return np.sqrt(np.diag(self.cov))

    @property
    def covariance(self):
        return np.linalg.inv(self.precision)

    @property
    def cov(self):
        return self.covariance

    def dist(self, dim0=0, dim1=None):
        """
        Exctracts a two-dimensional distribution MVN with the `dim0`th and
        `dim1`st component of this MVN.

        Instead if an index, you can also use the name of the parameter.
        """
        if isinstance(dim0, str):
            dim0 = self.index(dim0)
        if isinstance(dim1, str):
            dim1 = self.index(dim1)

        if dim1 is None:
            return norm(loc=self.mean[dim0], scale=self.cov[dim0, dim0] ** 0.5)
        else:
            dim = [dim0, dim1]
            dim_grid = np.ix_(dim, dim)
            sub_cov = self.cov[dim_grid]
            return multivariate_normal(mean=self.mean[dim], cov=sub_cov)

    def __str__(self):
        headers = ["name", "µ", "σ"]
        data = [self.parameter_names, self.mean, self.std_diag]
        data_T = list(zip(*data))
        s = tabulate(data_T, headers=headers)
        return s

    def __eq__(self, other):
        return (
            np.max(np.abs(self.mean - other.mean)) < 1.0e-12
            and np.max(np.abs(self.precision - other.precision)) < 1.0e-12
            and self.name == other.name
            and self.parameter_names == other.parameter_names
        )


class Gamma:
    def __init__(self, shape=1.0e-6, scale=1e6, name="Gamma"):
        """
        Creates a Gamma distribution from provided `shape` k and `scale` θ
        with the PDF

            f(x) = 1 / [Γ(k) θ^k] x^(k-1) exp(-x/θ)

        where Γ denotes the Gamma function.

        shape:
            shape parameter k

        scale:
            scale parameter θ

        name:
            name of the distribution, e.g. to indicate "prior" or "posterior"

        Notes:
            The default values aim at creating a non-informative gamma
            distribution. Ideally, we would have shape=0 and scale=inf, but
            that raises numerical issues. Other authors [Kerman, Jouni.
            "Neutral noninformative and informative conjugate beta and gamma
            prior distributions." Electronic Journal of Statistics 5 (2011):
            1450-1470.] propose Gamma(1/3, 0).
        """
        self.scale = scale
        self.shape = shape
        self.name = name

    @property
    def mean(self):
        return self.scale * self.shape

    @property
    def std(self):
        return self.scale * self.shape ** 0.5

    def dist(self):
        return gamma(a=self.shape, scale=self.scale)

    def __repr__(self):
        return f"{self.name:15} | mean:{self.mean:10.6f} | scale:{self.scale:10.6f} | shape:{self.shape:10.6f}"

    @classmethod
    def FromQuantiles(cls, x0, x1, p=(0.05, 0.95)):
        """
        Create a gamma distribution from the given quantiles such that
            gamma.cdf(x0) = p[0]
            gamma.cdf(x1) = p[1]
        following the approach from
        https://www.johndcook.com/quantiles_parameters.pdf (Chapter 4)
        """

        assert x0 < x1
        assert p[0] < p[1]
        _ppf = gamma.ppf

        # As the Gamma distribution is from the scale family, it follows that
        #   scale = x_i / PPF(p_i; shape,1)
        # which we can use to elimate the scale parameter:
        #       x0 / PPF(p0; shape) = x1 / PPF(p1; shape)
        # This equation is reformulated as a function of shape to find its root.

        def f(shape):
            return _ppf(p[1], shape) / _ppf(p[0], shape) - x1 / x0

        # As f is strictly monotonically decreasing, we efficiently find the
        # single root by first finding the bracket [c, F*c] such that
        #    f(c) < 0 and f(F*c) > 0 ...

        c, F = 61.74, 4.2  # Nothing up my sleeve numbers. Only influences performance.
        while f(c) < 0.0:
            c /= F
        while f(F * c) > 0.0:
            c *= F

        # ... and by then applying a root finding algorithm.
        shape = brenth(f, a=c, b=F * c, disp=True)

        scale = x0 / _ppf(q=p[0], a=shape)
        return cls(shape=shape, scale=scale)

    @classmethod
    def FromSDQuantiles(cls, sd0, sd1, p=(0.05, 0.95)):
        """
        In the context of VB, the gamma distribution is used to model the noise
        _precision_. In practice, it can be convenient, do generate this
        distribution from the standard deviation (SD)
        """
        assert sd0 < sd1
        return Gamma.FromQuantiles(1 / sd1 ** 2, 1 / sd0 ** 2, p)

    @classmethod
    def FromMeanStd(cls, mean, std):
        variance = std ** 2
        scale = variance / mean
        return cls(shape=mean / scale, scale=scale)
