
import pickle
import torch
from eqprop_torch import EqPropNet


class EqPropNet_NoGrad(EqPropNet):
    def __init__(self, *args, **kwargs):
        """
        An equilibrium propagation network that works on PyTorch
        without using gradients.
        """
        super().__init__(*args, **kwargs)

        # Define the derivative of rho
        self.rho_grad = lambda x: ((0 <= x) * (x <= 1)).type(torch.FloatTensor)


    def step(self, states, y=None):
        """
        Make one step of duration dt. TODO
        """
        rho = self.rho
        rho_grad = self.rho_grad

        # See XXX in self.energy() in eqprop_torch.py
        rho = lambda x: x
        rho_grad = lambda x: 1

        # Update states
        for i in range(1, len(states)):
            # Calculate the gradient ds/dt = -dE/ds
            states_grad = -states[i] + rho_grad(states[i]) * (
                rho(states[i-1]) @ self.weights[i-1] + self.biases[i-1])
            # Update and clamp
            states[i] += self.dt * states_grad
            states[i].clamp_(0,1)

        # Update last layer in weakly clamped phase
        if y is not None:
            states[-1] += self.dt * self.beta * (y - states[-1])
    

    def update_weights(self, free_states, clamped_states):
        """
        TODO
        """
        rho = self.rho

        # See XXX in self.energy() in eqprop_torch.py
        rho = lambda x: x

        for i in range(len(self.weights)):
            # Calculate weight gradient and update TODO: is there a faster way?
            weights_grad = \
                rho(clamped_states[i]).unsqueeze(dim=-1) \
                @ rho(clamped_states[i+1]).unsqueeze(dim=-2) \
                - rho(free_states[i]).unsqueeze(dim=-1) \
                @ rho(free_states[i+1]).unsqueeze(dim=-2)
            self.weights[i] += self.lr[i] / self.beta * weights_grad.mean(dim=0)

        for i in range(len(self.biases)):
            # Calculate weight gradient and update
            biases_grad = rho(clamped_states[i+1]) - rho(free_states[i+1])
            self.biases[i] += self.lr[i] / self.beta * biases_grad.mean(dim=0)



