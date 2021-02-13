from bayes.parameters import ModelParameters as ParameterList
from collections import OrderedDict
import numpy as np

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


class NormalPrior:
    def __init__(self, mean, sd):
        self.mean = mean
        self.sd = sd


class ModelError:
    def __call__(self, parameter_list):
        raise NotImplementedError("Override this!")


class LatentParameter:
    def __init__(self):
        self._affected_parameters = []
        self.N = None
        self.prior = None
        self.start_idx = None

    def _contains(self, key, name):
        return (key, name) in self._affected_parameters

    def add(self, key, name, N):
        assert not self._contains(key, name)
        if self.N is None:
            self.N = N
        else:
            assert self.N == N

        self._affected_parameters.append((key, name))

    def __len__(self):
        return len(self._affected_parameters)

    def __getitem__(self, i):
        return self._affected_parameters[i]

    def __str__(self):
        if self.N == 1:
            idx_range = str(self.start_idx)
        else:
            idx_range = f"{self.start_idx}..{self.start_idx+N}"

        return f"{idx_range:10} {self._affected_parameters} {self.prior}"


class LatentParameterList:
    def __init__(self):
        self._all_parameter_lists = {}
        self._latent = OrderedDict()
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
        
        if latent_name not in self._latent:
            self._latent[latent_name] = LatentParameter()

        self._latent[latent_name].add(key, parameter_name, N)
        self._update_idx()

    def add_by_name(self, latent_name):
        for key, prm_list in self._all_parameter_lists.items():
            if prm_list.has(latent_name):
                self.add(latent_name, latent_name, key)
        

    def _update_idx(self):
        self._total_length = 0
        for key, latent in self._latent.items():
            latent.start_idx = self._total_length
            self._total_length += latent.N or 0

    def update(self, number_vector):
        assert len(number_vector) == self._total_length
        for l in self._latent.values():
            latent_numbers = number_vector[l.start_idx : l.start_idx + l.N]
            for (key, prm_name) in l:
                self._all_parameter_lists[key][prm_name] = latent_numbers

    def __str__(self):
        s = ""
        for key, latent in self._latent.items():
            s += f"{key:10}: {latent} \n"
        return s


class Noise:
    def __init__(self):
        self._noise_groups = OrderedDict()
        self._noise_prior = OrderedDict()

    def add(self, noise_name, sensor, key=None):
        if noise_name not in self._noise_groups:
            self._noise_groups[noise_name] = []

        noise_group = self._noise_groups[noise_name]
        assert (key, sensor) not in noise_group

        for (_, existing_sensor) in noise_group:
            assert existing_sensor.shape == sensor.shape

        noise_group.append((key, sensor))


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
        sensors = self._sensor_data.keys()
        model_response = self._fw(prm, sensors, self._ts)
        error = {}
        for sensor in sensors:
            error[sensor] = model_response[sensor] - self._sensor_data[sensor]
        return error

class InferenceProblem:
    def __init__(self):
        self.latent = LatentParameterList()
        self.noise = Noise()
        self.model_errors = {}

    def add_model_error(self, model_error, parameter_list, key=None):
        key = key or len(self.model_errors)
        assert key not in self.model_errors

        self.model_errors[key] = model_error
        self.latent.add_parameter_list(parameter_list, key)

        return key

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

    def generate_data(N_time_steps, noise_sd):
        time_steps = np.linspace(0, 1, N_time_steps)
        model_response = fw(prm, [s1, s2, s3], time_steps)
        sensor_data = {}
        for sensor, perfect_data in model_response.items():
            sensor_data[sensor] = perfect_data + np.random.normal(0., noise_sd, N_time_steps) 
        return time_steps, sensor_data

    data1 = generate_data(101, noise_sd = 0.1)
    data2 = generate_data(51, noise_sd = 0.2)

    me1 = MyModelError(fw, data1)
    me2 = MyModelError(fw, data2)

    problem = InferenceProblem()
    key1 = problem.add_model_error(me1, fw.parameter_list())
    key2 = problem.add_model_error(me2, fw.parameter_list())

    problem.latent.add("A", "A", key1)
    problem.latent.add("A", "A", key2)
    problem.latent.add_by_name("B")

    print(problem.latent)





