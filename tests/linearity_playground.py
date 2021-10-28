import numpy as np
from bayes.vb import *
import matplotlib.pyplot as plt


class ModelError(VariationalBayesInterface):
    def __init__(self, A_true=5, order=1):
        self.x = np.r_[2, 3, 4, 5]
        self.order = order
        self.data = self.fw([A_true])

    def fw(self, A):
        return A[0] ** self.order * self.x

    def __call__(self, A):
        return {"n": self.fw(A) - self.data}

    def jacobian(self, A):
        dA = self.order * A ** (self.order - 1)
        return {"n": np.atleast_2d(dA * self.x).T}


class Linearized(VariationalBayesInterface):
    def __init__(self, me, MAP):
        self.x0 = me(MAP)
        self.MAP = MAP
        self.jac = me.jacobian(MAP)

    def __call__(self, A):
        return {"n": self.x0["n"] + self.jac["n"] @ (A[0] - self.MAP)}

    def jacobian(self, A):
        return self.jac


model_order = 4
A_true = 0.4 # looks bad, R² --> 0
A_true = 1.4 # looks OKish, R² --> 1
m = ModelError(A_true, order=model_order)

prior = MVN(0.9 * A_true, 1 / (0.2 * A_true) ** 2)
noise = Gamma.FromSDQuantiles(0.1, 0.2)

info = variational_bayes(m, prior, noise)

mlin = Linearized(m, info.param.mean)
info = variational_bayes(mlin, prior, noise)
mean, sd = info.means[-1], info.sds[-1] * 10

print(info)


positions = [mean + i * sd for i in np.linspace(-3, 3, 20)]
orig = np.array([m([x])["n"] for x in positions])
linear = np.array([mlin([x])["n"] for x in positions])

y_i = orig
f_i = linear

y_bar = np.mean(y_i, axis=0)

SS_res = np.sum((y_i - f_i) ** 2, axis=0)
SS_tot = np.sum((y_i - y_bar) ** 2, axis=0)

R2 = 1 - SS_res / SS_tot
print("R² values: ", R2)

plt.plot(positions, y_i, color="black", label="orig")
plt.plot(positions, f_i, color="red",label="linearized")

plt.legend()
plt.show()
