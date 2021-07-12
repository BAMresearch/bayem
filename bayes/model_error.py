# local imports
from .parameters import ParameterList
from .jacobian import d_model_error_d_named_parameter


class ModelErrorInterface:
    """
    The model error is understood here as the difference between the computed
    value of a forward model (say computed deflection by an FE-model) and the
    corresponding measured value (the measured deflection). This class is the
    base class for a problem-specific ModelError class. An object from such a
    class is equipped with a parameter_list, a forward model, and the capability
    to evaluate the model error and its derivative for a given parameter_list.
    """

    def __init__(self):
        # this parameter list will contain the parameters of the forward model,
        # both fixed and latent types
        self.parameter_list = ParameterList()

    def __call__(self):
        """
        Evaluates the model error based on 'self.parameter_list' as a dict of
        {key: numpy.array}. The keys are typically sensors here. Note that this
        method has to be overwritten, when defining a problem-specific model
        error using the problem-specific forward model.
        """
        raise NotImplementedError("Override this!")

    def jacobian(self, latent_names=None):
        """
        Computes the derivative of the model error with respect to the specified
        latent parameters.

        Parameters
        ----------
        latent_names : list, optional
            A list of strings with names of the latent parameters the derivative
            should be evaluated for. If not specified, the derivative will be
            evaluated for all latent parameters.

        Returns
        -------
        jac : dict
            Containing the derivatives of all model_errors with respect to the
            stated latent parameters.
        """
        jac = dict()
        latent_names = latent_names or self.parameter_list.names
        for prm_name in latent_names:
            # this is where the (partial) derivative is numerically evaluated
            prm_jac = d_model_error_d_named_parameter(self, prm_name)
            for key in prm_jac:
                if key not in jac:
                    jac[key] = dict()
                jac[key][prm_name] = prm_jac[key]
        return jac
