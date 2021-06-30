# ---------
# AUTHOR: Atul Agrawal (atul.agrawal@tum.de)
# ---------
"""
Interface to To interact with the black box forward solver. To override the pytorch autograd to pass grad obtained by the
adjoint method. The implementation assumes that the adjoint solver is implemented in the forward solver and the forward solver to have solve and adjoint methods implemented
ref: http://www.dolfin-adjoint.org/en/latest/documentation/maths/3-gradients.html

"""
import torch as th


class forwardSolverInterface:
    """
    Class for the blackbox solver
    # TODO : The forward solver should provide an operator which takes in functional(likelihood) and returns the adjoint based grad
    """
    def __init__(self, solver, known_input):
        """

        :param solver: Callable. Should have a method which takes in the functional and for the given solution,
        returns the grad by the adjoint mathod.
        :param functional: Callable. Returns the value of functional for a given input
        :param known_input [dict] :
        """
        self.solver = solver
        #self.functional = functional
        self.known_input = known_input

    def solve(self, latent_para):
        """
        :param latent_para [dict]: The parameters to be inferred
        :return:
        """
        u = self.solver.solve(self.known_input,latent_para) # pass the input as arguments here
        #grad = self.solver.Jacobian(self.functional,u)
        #grad = self.solver.Jacobian(u)
        return u
        #raise NotImplementedError("Implement me! Should return the grad and u")

    def override_autograd(self):

        class custom_autograd(th.autograd.Function):
            """
            Overrides the PyTorch autograd to include Jacobians coming from the forward solver
            """

            @staticmethod
            def forward(ctx, param):
                """

                :param ctx:
                :param param: Latent parameter to be inferred
                :return:
                """
                u = self.solve(param)
                u = th.tensor(u, requires_grad=True)

                # solver Jacobian, just an operator, not exactly a big chunky matrix
                #grad = th.tensor(grad)
                ctx.save_for_backward(state) # the state can be anything which a forward solver returns which maybe needed for grad computation later.

                return u

            @staticmethod
            def backward(ctx, grad_output):
                """
                x --> PDE --> y -- > L
                :param grad_output derivative of functional wrt y dL/dy, Eg posterior.backward()
                :return dLdx
                # TODO: Coordinate to implement custom adjoint solver
                """
                state = ctx.saved_tensors

                grad_input = self.solver.grad_adjoint(grad_output, state)
                return grad_input
        return custom_autograd.apply