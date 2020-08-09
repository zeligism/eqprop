
import os
import pickle
import torch
from math import sqrt


class EqPropNet:
    """
    This class implements Equilibrium Propagation.
    Paper: https://arxiv.org/abs/1602.05179
    Code: https://github.com/bscellier/Towards-a-Biologically-Plausible-Backprop
    """
    def __init__(self, batch_size, layer_sizes, learning_rates,
        free_iters, clamped_iters, beta=1, dt=0.5):
        """
        An equilibrium propagation network that works on PyTorch.
        """

        # Training-specific hyperparameters
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes
        self.lr = learning_rates
        self.free_iters = free_iters
        self.clamped_iters = clamped_iters
        # Dynamics-specific hyperparameters
        self.beta = beta
        self.dt = dt

        # Initialize states to 0 (or I guess anything between 0 and 1 is ok)
        self.states = [torch.zeros(batch_size, l) for l in layer_sizes]
        # Initialize weights using Glorot-Bengio initialization
        self.weights = [torch.randn(l1, l2) * sqrt(2. / (l1 + l2))
            for l1, l2 in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [torch.zeros(l) for l in layer_sizes]

    def rho(self, x):
        return torch.clamp(x,0,1)

    def energy(self, states):
        """
        Calculates the energy of the network.
        """

        # Sum of s_i * s_i for all i
        states_energy = 0.5 * sum(torch.sum(s*s, dim=1) for s in states)
        # Sum of W_ij * rho(s_i) * rho(s_j) for all i, j, i != j
        weights_energy = sum(
            # ((B, s_i) @ (s_i, s_j)) * (B, s_j) = (B, s_j) * (B, s_j)
            torch.sum(self.rho(s_i) @ W_ij * self.rho(s_j), dim=1)
            for W_ij, s_i, s_j in zip(self.weights, states, states[1:])
        )
        # Sum of bias_i * rho(s_i)
        biases_energy = sum(
            torch.sum(b_i * self.rho(s_i), dim=1)
            for b_i, s_i in zip(self.biases, states)
        )

        return states_energy - weights_energy - biases_energy


    def cost(self, states, y):
        """
        Calculates the cost between the state of the last layer of the network
        with the output y. The cost is just the distance (L2 loss).
        """
        cost = torch.sum((self.output_state(states) - y)**2, dim=-1)

        return cost


    def output_state(self, states=None):
        """
        Returns the output state layer from states.
        In our case, it is simply the last layer in `states`.
        """
        return self.states[-1] if states is None else states[-1]


    def clamp_input(self, x):
        """
        The following function simply clamps an input to the network.
        The input x should always be clamped to the first layer because
        our training procedure assumes that `self.states[0] == x`.
        """
        self.states[0] = x


    def step(self, states, y=None):
        """
        Make one step of duration dt.
        """

        [s.requires_grad_() for s in states[1:]]

        # Calculate the total energy with the cost if y is given
        energy = self.energy(states)
        if y is not None:
            energy += self.beta * self.cost(states, y)

        # Calculate the gradients (notice the negative sign because ds/dt = -dE/ds)
        (-energy).sum().backward()

        # Update states
        for i in range(1, len(states)):
            states[i] = states[i] + self.dt * states[i].grad
            states[i] = states[i].clamp(0,1).detach()

        return states


    def update_weights(self, free_states, clamped_states):
        """
        Updates weights along its gradient descent.
        """

        [w.requires_grad_() for w in self.weights]
        [b.requires_grad_() for b in self.biases]

        free_energy = self.energy(free_states)
        clamped_energy = self.energy(clamped_states)
        energy = (clamped_energy - free_energy) / self.beta
        (-energy).mean().backward()

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + self.lr[i] * self.weights[i].grad
            self.weights[i].detach_()
            self.biases[i+1] = self.biases[i+1] + self.lr[i] * self.biases[i+1].grad
            self.biases[i+1].detach_()


    def eqprop(self, x, y, train=True):
        """
        Trains the network on one example (x,y) using equilibrium propagation.
        """

        # First clamp the input
        self.clamp_input(x)

        # Run free phase
        for _ in range(self.free_iters):
            self.step(self.states)

        if train:
            # Collect states and perturb them to the weakly clamped y
            clamped_states = [s.clone().detach() for s in self.states]
            for _ in range(self.clamped_iters):
                self.step(clamped_states, y)

            # Update weights
            self.update_weights(self.states, clamped_states)

        with torch.no_grad():
            return self.energy(self.states), self.cost(self.states, y)


    def save_parameters(self, fname, models_dir="models"):
        """
        Saves the weights and biases of the model to a file called `fname`.
        """
        if not os.path.exists(models_dir):
            os.mkdir(models_dir)

        with open(os.path.join(models_dir, fname), "wb") as f:
            print("Saving parameters to '%s'... " % fname, end="")
            parameters = (self.weights, self.biases)
            pickle.dump(parameters, f)
            print("Done.")


    def load_parameters(self, fname):
        """
        Loads the weights and biases from a file called `fname`.
        """
        with open(fname, "rb") as f:
            print("Loading parameters from '%s'... " % fname, end="")
            parameters = pickle.load(f)
            self.weights, self.biases = parameters
            print("Done.")


class EqPropNet_NoGrad(EqPropNet):
    def __init__(self, *args, **kwargs):
        """
        An equilibrium propagation network that works on PyTorch
        without using gradients.
        """
        super().__init__(*args, **kwargs)

    def rho_grad(self, x):
        """
        Define the derivative of rho.
        """
        return ((0 <= x) * (x <= 1)).to(x)

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
            states[i] = (states[i] + self.dt * states_grad).clamp(0,1)

        # Update last layer in weakly clamped phase
        if y is not None:
            output_state_grad = self.beta * (y - self.output_state())
            states[-1] = (states[-1] + self.dt * output_state_grad).clamp(0,1)

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


class ContEqPropNet(EqPropNet):
    """
    This class implements C-EqProp, aka EqProp with continual weight updates.
    Paper: https://arxiv.org/abs/2005.04168.pdf
    Code: https://drive.google.com/open?id=1oZtzBTu8zZgvAopyK2sQg2bpcsrzwTrp
    (Author: "HEY GUYS I HAVE A GREAT IDEA LET'S SHARE OUR CODE THROUGH GOOGLE DRIVE.")
    """

    def eqprop(self, x, y, train=True):
        """
        Trains the network on one example (x,y) using equilibrium propagation
        with continual weight updates.
        """

        # First clamp the input
        self.clamp_input(x)

        # Run free phase
        for _ in range(self.free_iters):
            self.step(self.states)

        if train:
            # Collect states and perturb them to the weakly clamped y
            states = [s.clone().detach() for s in self.states]
            for _ in range(self.clamped_iters):
                # Copy current states at time t
                prev_states = [s.clone() for s in states]
                # Do one eqprop step on states
                self.step(states, y)
                # Update weights based on a perturbation from states(t) to states(t+1)
                # so states(t+1) is clamped and states(t) is free, in some sense.
                self.update_weights(prev_states, states)

        with torch.no_grad():
            return self.energy(self.states), self.cost(self.states, y)


class ContEqPropNet_NoGrad(ContEqPropNet, EqPropNet_NoGrad):
    """
    EqProp with continual weight updates without using autograd.
    """
    def eqprop(self, *args, **kwargs):
        with torch.no_grad():
            return super().eqprop(*args, **kwargs)


