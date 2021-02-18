from bayes.parameters import *
from bayes.inference_problem import *
import itertools

import scipy.optimize

class Sensor:
    def __init__(self, name, shape=(1, 1)):
        self.name = name
        self.shape = shape

    def __repr__(self):
        return f"{self.name} {self.shape}"

class MySensor(Sensor):
    def __init__(self, name, position):
        super().__init__(name, shape=(1, 1))
        self.position = position


class MyForwardModel:
    def __call__(self, parameter_list, sensors, time_steps):
        """
        evaluates 
            fw(x, t) = A * x + B * t
        """
        A = parameter_list["A"]
        B = parameter_list["B"]

        result = {}
        for sensor in sensors:
            result[sensor] = A * sensor.position + B * time_steps
        return result

    def parameter_list(self):
        p = ModelErrorParameters()
        p.define("A", None)
        p.define("B", None)
        return p


class MyModelError(ModelError):
    def __init__(self, fw, data):
        self._fw = fw
        self._ts, self._sensor_data = data

    def __call__(self, prm):
        sensors = list(self._sensor_data.keys())
        model_response = self._fw(prm, sensors, self._ts)
        error = {}
        for sensor in sensors:
            error[sensor] = model_response[sensor] - self._sensor_data[sensor]
        return error


"""
###############################################################################

                            MAIN CODE

###############################################################################
"""

if __name__ == "__main__":
    # Define the sensor
    s1, s2, s3 = MySensor("S1", 0.2), MySensor("S2", 0.5), MySensor("S3", 42.0)

    fw = MyForwardModel()
    prm = fw.parameter_list()

    # set the correct values
    A_correct = 42.0
    B_correct = 6174.0
    prm["A"] = A_correct
    prm["B"] = B_correct

    np.random.seed(6174)
    noise_sd1 = 0.2
    noise_sd2 = 0.4

    def generate_data(N_time_steps, noise_sd):
        time_steps = np.linspace(0, 1, N_time_steps)
        model_response = fw(prm, [s1, s2, s3], time_steps)
        sensor_data = {}
        for sensor, perfect_data in model_response.items():
            sensor_data[sensor] = perfect_data + np.random.normal(
                0.0, noise_sd, N_time_steps
            )
        return time_steps, sensor_data

    data1 = generate_data(101, noise_sd1)
    data2 = generate_data(51, noise_sd2)

    me1 = MyModelError(fw, data1)
    me2 = MyModelError(fw, data2)

    problem = VariationalBayesProblem()
    key1 = problem.add_model_error(me1, fw.parameter_list())
    key2 = problem.add_model_error(me2, fw.parameter_list())

    problem.latent.add("A", "A", key1)
    problem.latent.add("A", "A", key2)
    problem.latent.add_by_name("B")

    problem.set_normal_prior("A", 40., 5.)
    problem.set_normal_prior("B", 6000., 300.)
    
    problem.define_noise_group("prec_exp1", 3*noise_sd1)
    problem.define_noise_group("prec_exp2", 3*noise_sd2)
    
    problem.add_to_noise_group("prec_exp1", s1, key1)
    problem.add_to_noise_group("prec_exp1", s2, key1)
    problem.add_to_noise_group("prec_exp1", s3, key1)

    problem.add_to_noise_group("prec_exp2", s1, key2)
    problem.add_to_noise_group("prec_exp2", s2, key2)
    problem.add_to_noise_group("prec_exp2", s3, key2)

    print(problem.latent)

    info = problem.run()

    # We now transform the vb problem into a sampling problem. 
    # 1) Define an additional parameter list for the noise hyperparameters
    #    and add it

    hyper_prm, hyper_key = ModelErrorParameters(), "hyper"
    problem.latent.define_parameter_list(hyper_prm, hyper_key)
    for name in problem.noise_groups:
        hyper_prm.define(name)
        problem.latent.add(name, name, hyper_key)

    print(problem.latent)

    # 2) Define a log likelihood function based on the noise groups
    def log_like_f(number_vector):
        me_by_noise = problem(number_vector)

        log_like = 0.0
        for noise_name, error in me_by_noise.items():
            precision = hyper_prm[noise_name]
            log_like = log_like - 0.5 * (
                len(error) * np.log(2.0 * np.pi / precision)
                + np.sum(np.square(error * precision))
            )
        return log_like

    # 3) Define prior distributions in a tool of your choice!
    import pymc3 as pm
    import theano.tensor as tt
    
    class LogLike(tt.Op):
        itypes = [tt.dvector]  # expects a vector of parameter values when called
        otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)
        def __init__(self, loglike):
            self.likelihood = loglike

        def perform(self, node, inputs, outputs):
            theta, = inputs  # this will contain my variables
            result = self.likelihood(theta)
            outputs[0][0] = np.array(result)  # output the log-likelihood

    pymc3_log_like = LogLike(log_like_f)
    pymc3_prior = [None] * len(problem.latent)

    with pm.Model() as model:
        for name, latent in problem.latent.items():
            idx = latent.start_idx
            assert latent.N == 1 # vector parameters not yet supported!
            try:
                mean, sd = latent.vb_prior
                # It is a parameter prior normal prior
                pymc3_prior[idx] = pm.Normal(name, mu=mean, sigma=sd)
            except AttributeError:
                # it is a noise prior
                gamma = problem.noise_groups[name].gamma
                s, c = gamma.s[0], gamma.c[0]
                alpha, beta = s, 1./c
                print(alpha, beta, alpha/beta)
                pymc3_prior[idx] = pm.Gamma(name, alpha=alpha, beta=beta)
                print(gamma)

        theta = tt.as_tensor_variable(pymc3_prior)
        pm.Potential("likelihood", pymc3_log_like(theta))

        # trace = pm.sample()
        trace = pm.sample(
                draws=500,
                step=pm.Metropolis(),
                chains=4,
                tune=100, 
                discard_tuned_samples=True,
            )
    summary = pm.summary(trace)
    print(summary)

    print(1./info.noise.mean[0]**0.5, 1./info.noise.mean[1]**0.5)
   
    means = summary["mean"]
    print(1./means["prec_exp1"]**0.5, 1./means["prec_exp2"]**0.5)

