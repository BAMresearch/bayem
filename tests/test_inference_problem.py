from bayes.parameters import ModelParameters as ParameterList
from collections import OrderedDict
import numpy as np

from bayes.vb import *
import itertools

import scipy.optimize

"""
###############################################################################

                            LIBRARY CODE

###############################################################################
"""


class Sensor:
    def __init__(self, name, shape=(1, 1)):
        self.name = name
        self.shape = shape

    def __repr__(self):
        return f"{self.name} {self.shape}"


class DummySensor(Sensor):
    def __init__(self):
        super().__init__("Dummy")


class NormalDistribution:
    def __init__(self, mean, sd):
        self.mean = mean
        self.sd = sd

    def __str__(self):
        return f"N({self.mean}, {self.sd})"


class ModelError:
    def __call__(self, parameter_list):
        raise NotImplementedError("Override this!")


class LatentParameter(list):
    def __init__(self):
        self.N = None
        self.prior = None
        self.start_idx = None

    def add(self, key, name, N):
        assert not (key, name) in self
        if self.N is None:
            self.N = N
        else:
            assert self.N == N

        self.append((key, name))

    def __str__(self):
        if self.N == 1:
            idx_range = str(self.start_idx)
        else:
            idx_range = f"{self.start_idx}..{self.start_idx+N}"

        return f"{idx_range:10} {list.__str__(self)} {self.prior}"

    def extract(self, all_numbers):
        if self.N == 1:
            return all_numbers[self.start_idx]
        else:
            return all_numbers[self.start_idx : self.start_idx + self.N]


class LatentParameterList(OrderedDict):
    def __init__(self):
        self._all_parameter_lists = {}
        self._total_length = None

    def add_parameter_list(self, parameter_list, key=None):
        assert key not in self._all_parameter_lists
        self._all_parameter_lists[key] = parameter_list

    def add(self, latent_name, parameter_name, key=None):
        assert key in self._all_parameter_lists
        assert self._all_parameter_lists[key].has(parameter_name)

        try:
            N = len(self._all_parameter_lists[key][parameter_name])
        except:
            N = 1

        if latent_name not in self:
            self[latent_name] = LatentParameter()

        self[latent_name].add(key, parameter_name, N)
        self._update_idx()

    def add_by_name(self, latent_name):
        for key, prm_list in self._all_parameter_lists.items():
            if prm_list.has(latent_name):
                self.add(latent_name, latent_name, key)

    def set_prior(self, latent_name, prior):
        assert self[latent_name].prior is None
        self[latent_name].prior = prior

    def _update_idx(self):
        self._total_length = 0
        for key, latent in self.items():
            latent.start_idx = self._total_length
            self._total_length += latent.N or 0

    def update(self, number_vector):
        assert len(number_vector) == self._total_length
        for l in self.values():
            latent_numbers = l.extract(number_vector)
            for (key, prm_name) in l:
                self._all_parameter_lists[key][prm_name] = latent_numbers
        return self._all_parameter_lists

    def __str__(self):
        s = ""
        for key, latent in self.items():
            s += f"{key:10}: {latent} \n"
        return s


class SensorGroup(set):
    def __init__(self, noise_model):
        self.noise_model = noise_model

    def add_sensor(self, sensor, key):
        if (sensor, key) in self:
            print("Whow. You already added that. Next time, this is an error!")
        for (existing_sensor, _) in self:
            assert existing_sensor.shape == sensor.shape
        self.add((sensor, key))

class SensorGroups(OrderedDict):
    """
    Dict {name : set(model_error_key, sensor)} to group the outputs
    of possibly multiple model errors. 
    """

    def define(self, name, noise_model):
        assert name not in self
        self[name] = SensorGroup(noise_model)
        return name # better: NoiseKey(name)

    def add(self, name, sensor, key=None):
        assert name in self
        self[name].add_sensor(sensor, key)


class NoiseModel:
    """
    Maybe even some "evaluate" here...
    """
    def __init__(self, sensor, key=None):
        self.sensor = sensor
        self.key = key

    def define_parameter_list(self):
        raise NotImplementedError()

class UncorrelatedNoise(NoiseModel):
    def define_parameter_list(self):
        p = ParameterList()
        p.define("sigma") 
        return p

def is_noise_key(key):
    return key.startswith("noise")

class InferenceProblem:
    def __init__(self):
        self.latent = LatentParameterList()
        self.noise_models = {}
        self.model_errors = {}

    def add_model_error(self, model_error, parameter_list, key=None):
        key = key or len(self.model_errors)
        assert key not in self.model_errors

        self.model_errors[key] = model_error
        self.latent.add_parameter_list(parameter_list, key)

        return key

    def add_noise_model(self, noise_model, parameter_list, noise_key=None):
        noise_key = noise_key or "noise"+str(len(self.noise_models))
        assert noise_key not in self.noise_models
        assert is_noise_key(noise_key)

        self.noise_models[noise_key] = noise_model
        self.latent.add_parameter_list(parameter_list, noise_key)

        return noise_key

    def add_sensor_to_noise(self, noise_key, sensor, key=None):
        assert key in self.model_errors
        assert noise_key in self.noise_models

        sensors = self.noise_models[noise_key].sensors
        if (key, sensor) in sensors:
            print(f"{key, sensor} already in group {noise_key}! Last warning!")
        sensors.add((key, sensor))


    def __call__(self, number_vector):
        prm_lists = self.latent.update(number_vector)
        result = {}
        for key, me in self.model_errors.items():
            result[key] = me(prm_lists[key])
        return result

    def eval_by_noise_groups(self, number_vector):
        model_response = self(number_vector)
        errors_by_noise = OrderedDict()
        for noise_name, noise in self.noise.items():
            errors = []
            for (key, sensor) in noise:
                errors.append(model_response[key][sensor])
            errors_by_noise[noise_name] = np.concatenate(errors)
        return errors_by_noise


class VariationalProblem:
    def __init__(self, inference_problem):
        self.inference_problem = inference_problem
        self.parameter_prior = self.prior_MVN()
        self.noise_prior = self.prior_noise()
        self._build_noise_pattern()
        # self.noise_pattern = None

    def prior_MVN(self):
        from bayes.vb import MVN

        means = []
        precs = []

        for latent in self.inference_problem.latent.values():
            assert type(latent.prior) == NormalDistribution
            for _ in range(latent.N):
                means.append(latent.prior.mean)
                precs.append(1.0 / latent.prior.sd ** 2)

        return MVN(means, np.diag(precs))

    def prior_noise(self):
        from bayes.vb import Gamma

        scales = []
        shapes = []
        for noise in self.inference_problem.noise.values():
            prior = noise.prior
            assert type(prior) == Gamma
            assert len(prior.s) == 1
            scales.append(prior.c[0])
            shapes.append(prior.s[0])

        return Gamma(c=scales, s=shapes)

    def _build_noise_pattern(self):
        # evaluate problem once to get the shapes
        errors_by_noise = self.inference_problem.eval_by_noise_groups(
            self.parameter_prior.mean
        )
        self.noise_pattern = []
        i = 0
        for error in errors_by_noise.values():
            N = len(error)
            self.noise_pattern.append(list(range(i, i + N)))
            i += N

    def __call__(self, number_vector):
        errors_by_noise = self.inference_problem.eval_by_noise_groups(number_vector)
        return np.concatenate(list(errors_by_noise.values()))


class PyMC3Problem:
    def __init__(self, inference_problem):
        self.inference_problem = inference_problem

    def __call__(self, number_and_noise_vector):
        """just builds the likelihood f"""
        N_noise = len(self.inference_problem.noise)
        number_vector = number_and_noise_vector[:-N_noise]
        noise_vector = number_and_noise_vector[-N_noise:]

        errors_by_noise = self.inference_problem.eval_by_noise_groups(number_vector)

        log_like = 0.0
        for error, sigma in zip(errors_by_noise.values(), noise_vector):
            log_like = log_like - 0.5 * (
                len(error) * np.log(2.0 * np.pi * (sigma ** 2))
                + np.sum(np.square(error / sigma ** 2))
            )

        return log_like


"""
###############################################################################

                            USER CODE

###############################################################################
"""


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
        p = ParameterList()
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

    np.random.seed = 6174
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

    problem = InferenceProblem()
    key1 = problem.add_model_error(me1, fw.parameter_list())
    key2 = problem.add_model_error(me2, fw.parameter_list())

    problem.latent.add("A", "A", key1)
    problem.latent.add("A", "A", key2)
    problem.latent.add_by_name("B")

    problem.latent.set_prior("A", NormalDistribution(40, 5))
    problem.latent.set_prior("B", NormalDistribution(6000, 300))

    tmp = problem([A_correct, B_correct])

    for key in [key1, key2]:
        noise_prm_name = "sigma" + str(key)
        for sensor in [s1, s2, s3]:
            noise_model = UncorrelatedNoise(key, sensor)
            noise_prm = noise_model.define_parameter_list()
            noise_key = problem.add_noise_model(noise_model, noise_prm)

            problem.latent.add(noise_prm_name, "sigma", noise_key)
    
    problem.latent.set_prior("sigma"+str(key1), Gamma.FromSD(3 * noise_sd2))
    problem.latent.set_prior("sigma"+str(key2), Gamma.FromSD(3 * noise_sd1))

    print(problem.latent)
    exit()
    


    vb_problem = VariationalProblem(problem)
    print(vb_problem.prior_MVN())

    print(problem.noise)
    print(vb_problem.prior_noise())

    # print(vb_problem([A_correct, B_correct]))

    info = variational_bayes(
        vb_problem, vb_problem.parameter_prior, vb_problem.noise_prior
    )
    print(info)

    print(1.0 / info.noise.mean[0] ** 0.5)
    print(1.0 / info.noise.mean[1] ** 0.5)

    pymc3_problem = PyMC3Problem(problem)
    print(pymc3_problem([0, 1, 2, 3]))

    def max_likelihood(numbers):
        return -pymc3_problem(numbers)

    result = scipy.optimize.minimize(
        max_likelihood,
        x0=[A_correct, B_correct, noise_sd1, noise_sd2],
        method="Nelder-Mead",
    )
    print(result)
