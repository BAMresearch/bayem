import numpy as np
import scipy.stats
from tabulate import tabulate


class MVN:
    def __init__(self, mean=[0.0], precision=[[1.0]], name="MVN", parameter_names=None):
        self.mean = np.atleast_1d(mean).astype(float)
        self.precision = np.atleast_2d(precision).astype(float)
        self.name = name
        if parameter_names is None:
            self.parameter_names = [f"p{i}" for i in range(len(self.mean))]
        else:
            self.parameter_names = parameter_names

        assert len(self.mean) == len(self.precision)

        if self.parameter_names is not None:
            assert len(self.parameter_names) == len(self.mean)

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
        """
        if dim1 is None:
            return scipy.stats.norm(
                loc=self.mean[dim0], scale=self.cov[dim0, dim0] ** 0.5
            )
        else:
            dim = [dim0, dim1]
            dim_grid = np.ix_(dim, dim)
            sub_cov = self.cov[dim_grid]
            return scipy.stats.multivariate_normal(mean=self.mean[dim], cov=sub_cov)

    def __str__(self):
        headers = ["name", "µ", "σ"]
        data = [self.parameter_names, self.mean, self.std_diag]
        data_T = list(zip(*data))
        s = tabulate(data_T, headers=headers)
        return s


class Gamma:
    def __init__(self, shape=1.0, scale=1.0, name="Gamma"):
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
        return scipy.stats.gamma(a=self.shape, scale=self.scale)

    def __repr__(self):
        return f"{self.name:15} | mean:{self.mean:10.6f} | scale:{self.scale:10.6f} | shape:{self.shape:10.6f}"

    @classmethod
    def FromSD(cls, sd, shape=1.0):
        return cls(shape, 1.0 / sd ** 2 / shape)

    @classmethod
    def Noninformative(cls):
        """
        Suggested by @ilma following
        https://projecteuclid.org/euclid.ejs/1320416981
        or
        https://math.stackexchange.com/questions/449234/vague-gamma-prior
        """
        return cls(scale=1.0 / 3.0, shape=0.0)
