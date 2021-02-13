from bayes.parameters import ModelParameters as ParameterList
from collections import OrderedDict
import numpy as np

from bayes.vb import *

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

        return f"{idx_range:10} {self._affected_parameters} {self.prior}"

    def extract(self, all_numbers):
        if self.N == 1:
            return all_numbers[self.start_idx]
        else:
            return all_numbers[self.start_idx:self.start_idx+self.N]

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

class NoiseGroup(list):
    def __init__(self):
        self.prior = None

    def add(self, sensor, key):
        assert (key, sensor) not in self
        for (_, existing_sensor) in self:
            assert existing_sensor.shape == sensor.shape
        self.append((key, sensor))

class NoiseGroups(OrderedDict):
    def add(self, noise_name, sensor, key=None):
        if noise_name not in self:
            self[noise_name] = NoiseGroup()

        self[noise_name].add(sensor, key)

    def set_prior(self, noise_name, prior):
        assert noise_name in self
        assert self[noise_name].prior is None
        self[noise_name].prior = prior

class InferenceProblem:
    def __init__(self):
        self.latent = LatentParameterList()
        self.noise = NoiseGroups()
        self.model_errors = {}

    def add_model_error(self, model_error, parameter_list, key=None):
        key = key or len(self.model_errors)
        assert key not in self.model_errors

        self.model_errors[key] = model_error
        self.latent.add_parameter_list(parameter_list, key)

        return key

    def __call__(self, number_vector):
        prm_lists = self.latent.update(number_vector)
        result = {}
        for key, me in self.model_errors.items():
            result[key] = me(prm_lists[key])
        return result

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
                precs.append(1./latent.prior.sd**2)
        
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
        errors_by_noise = self.eval_by_noise_groups(self.parameter_prior.mean)
        self.noise_pattern = []
        i = 0
        for error in errors_by_noise.values():
            N = len(error)
            self.noise_pattern.append(list(range(i, i+N)))
            i += N

    def eval_by_noise_groups(self, number_vector):
        model_response = self.inference_problem(number_vector)
        errors_by_noise = OrderedDict()
        for noise_name, noise in self.inference_problem.noise.items():
            errors = []
            for (key, sensor) in noise:
                errors.append(model_response[key][sensor])
            errors_by_noise[noise_name] = np.concatenate(errors)
        return errors_by_noise

    def __call__(self, number_vector):
        return np.concatenate(list(self.eval_by_noise_groups(number_vector).values()))


class PyMC3Problem:
    def __init__(self, inference_problem):
        self.inference_problem = inference_problem

"""
###############################################################################

                            USER CODE

###############################################################################
"""

class MySensor(Sensor):
    def __init__(self, name, position):
        super().__init__(name, shape=(1,1))
        self.position = position

class MyForwardModel:
    def __call__(self,parameter_list, sensors, time_steps):
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
    s1, s2, s3 = MySensor("S1", 0.2), MySensor("S2", 0.5), MySensor("S3", 42.)

    fw = MyForwardModel()
    prm = fw.parameter_list()
    
    # set the correct values
    A_correct = 42.
    B_correct = 6174.
    prm["A"] = A_correct
    prm["B"] = B_correct

    np.random.seed= 6174
    noise_sd1 = 0.2
    noise_sd2 = 0.4

    def generate_data(N_time_steps, noise_sd):
        time_steps = np.linspace(0, 1, N_time_steps)
        model_response = fw(prm, [s1, s2, s3], time_steps)
        sensor_data = {}
        for sensor, perfect_data in model_response.items():
            sensor_data[sensor] = perfect_data + np.random.normal(0., noise_sd, N_time_steps) 
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

    problem.noise.add("exp1_noise", s1, key1)
    problem.noise.add("exp1_noise", s2, key1)
    problem.noise.add("exp1_noise", s3, key1)
    problem.noise.set_prior("exp1_noise", Gamma.FromSD(3 * noise_sd1))

    problem.noise.add("exp2_noise", s1, key2)
    problem.noise.add("exp2_noise", s2, key2)
    problem.noise.add("exp2_noise", s3, key2)
    problem.noise.set_prior("exp2_noise", Gamma.FromSD(3* noise_sd2))

    vb_problem = VariationalProblem(problem)
    print(vb_problem.prior_MVN())

    print(problem.noise)
    print(vb_problem.prior_noise())
    
    # print(vb_problem([A_correct, B_correct]))

    info = variational_bayes(vb_problem, vb_problem.parameter_prior, vb_problem.noise_prior)
    print(info)

    print(1. / info.noise.mean[0]**0.5)
    print(1. / info.noise.mean[1]**0.5)

