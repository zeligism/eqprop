
import os
import pickle
import torch
from math import sqrt


class EqPropNet:
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



