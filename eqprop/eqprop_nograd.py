
import torch
from .eqprop import EqPropNet


class EqPropNet_NoGrad(EqPropNet):
    def __init__(self, *args, **kwargs):
        """
        An equilibrium propagation network that works on PyTorch
        without using gradients.
        """
        super().__init__(*args, **kwargs)

        # Define the derivative of rho
        self.rho_grad = lambda x: ((0 <= x) * (x <= 1)).type(torch.FloatTensor)

    def eqprop(self, *args, **kwargs):
        with torch.no_grad():
            return super().eqprop(*args, **kwargs)

    def step(self, states, y=None):
        """
        Make one step of duration dt.
        """
        # Update states
        for i in range(1, len(states)):
            # Calculate inner gradient terms
            norm_grad = -states[i]
            forward_grad = self.rho(states[i-1]) @ self.weights[i-1] if i > 0 else 0
            backward_grad = self.rho(states[i+1]) @ self.weights[i].t() if i < len(states) - 1 else 0
            bias_grad = self.biases[i]
            
            # Calculate the whole gradient ds/dt = -dE/ds
            states_grad = norm_grad + self.rho_grad(states[i]) * (forward_grad + backward_grad + bias_grad)

            # Update and clamp
            states[i] = self.rho(states[i] + self.dt * states_grad)

        # Update last layer in weakly clamped phase
        if y is not None:
            output_state_grad = self.beta * (y - self.output_state())
            states[-1] = self.rho(states[-1] + self.dt * output_state_grad)

        return states
    

    def update_weights(self, free_states, clamped_states):
        """
        Updates weights based on eqprop dynamics.
        """

        for i in range(len(self.weights)):
            # Calculate weight gradient and update TODO: is there a faster way?
            weights_grad = \
                self.rho(clamped_states[i]).unsqueeze(dim=2) \
                @ self.rho(clamped_states[i+1]).unsqueeze(dim=1) \
                - self.rho(free_states[i]).unsqueeze(dim=2) \
                @ self.rho(free_states[i+1]).unsqueeze(dim=1)
            self.weights[i] += self.lr[i] / self.beta * weights_grad.mean(dim=0)

        for i in range(1, len(self.biases)):
            # Calculate weight gradient and update
            biases_grad = self.rho(clamped_states[i]) - self.rho(free_states[i])
            self.biases[i] += self.lr[i-1] / self.beta * biases_grad.mean(dim=0)


