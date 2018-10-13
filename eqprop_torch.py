
import pickle
import torch

from training_configs import *


class EqPropNet:
    def __init__(self, layers_sizes=LAYER_SIZES,
        learning_rates=LEARNING_RATES, beta=BETA, rho=RHO):
        """
        An equilibrium propagation network that works on PyTorch.
        """
        self.layers_sizes = layers_sizes
        self.learning_rates = learning_rates
        self.beta = beta
        self.rho = rho

        # Lists storing the weights, biases, and neuron-states of each layer
        self.W = [torch.randn(l1, l2) * torch.tensor(2 / (l1 + l2)).sqrt()
            for l1, l2 in zip(layers_sizes[:-1], layers_sizes[1:])]
        self.biases = [torch.zeros(l) for l in layers_sizes[1:]]
        self.states = [torch.zeros(BATCH_SIZE, l) for l in layers_sizes]

    
    def energy(self, states):
        """
        Calculates the energy of the network.
        """
        rho = self.rho
        energy = 0

        for i in range(len(states)):
            # Sum of s_i * s_i for all i
            energy += 0.5 * torch.sum(states[i] * states[i], dim=-1)

        for i in range(len(self.W)):
            # Sum of W_ij * rho(s_i) * rho(s_j) for all i, j
            energy -= torch.sum(
                (rho(states[i]) @ self.W[i]) * rho(states[i+1]), dim=-1)
            # Sum of bias_i * rho(s_i)
            energy -= torch.sum(self.biases[i] * rho(states[i+1]), dim=-1)

        return energy


    def cost(self, states, y):
        """
        Calculates the cost between the state of the last layer of the network
        with the output y. The cost is just the distance (L2 loss).
        """
        output_layer = self.output_state(states)
        cost = torch.sum((output_layer - y) ** 2, dim=-1)
        
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


    def step(self, states, y=None, dt=DELTA):
        """
        Make one step of duration dt. TODO
        """

        [s.requires_grad_() for s in states[1:]]

        # Calculate the total energy with the cost if y is given
        energy = self.energy(states)
        if y is not None:
            energy += self.beta * self.cost(states, y)

        # Calculate the gradients
        energy.sum().backward()

        # Update states
        for i in range(1, len(states)):
            # Notice the negative sign because ds/dt = -dE/ds (partial d)
            states[i] = states[i] - dt * states[i].grad
            states[i].clamp_(0,1).detach_()

        return energy


    def eqprop(self, x, y, train=True):
        """
        Trains the network on one example (x,y) using equilibrium propagation.
        """

        # First clamp the input
        self.clamp_input(x)

        # Run free phase
        for i in range(N_ITER_1):
            energy = self.step(self.states)

        if train:
            # Collect states and perturb them to the weakly clamped y
            clamped_states = [torch.tensor(s) for s in self.states]
            for i in range(N_ITER_2):
                energy = self.step(clamped_states, y)

            # Update weights
            self.update_weights(self.states, clamped_states)

        return energy, self.cost(self.states, y)
    

    def update_weights(self, free_states, clamped_states, dt=DELTA):
        """
        TODO: doc
        """
        lr = self.learning_rates

        [w.requires_grad_() for w in self.W]
        [b.requires_grad_() for b in self.biases]

        free_energy = self.energy(free_states).mean()
        clamped_energy = self.energy(clamped_states).mean()
        energy = 1 / self.beta * (clamped_energy - free_energy)
        energy.backward()

        # Update weights and biases (note the negative sign)
        for i in range(len(self.W)):
            self.W[i] = self.W[i] - lr[i] * self.W[i].grad
            self.W[i].detach_()

        for i in range(len(self.biases)):
            self.biases[i] = self.biases[i] - lr[i] * self.biases[i].grad
            self.biases[i].detach_()


    def save_parameters(self, fname=FNAME):
        """
        Saves the weights and biases of the model to a file called `fname`.
        """
        with open(fname, "wb") as f:
            print("Saving parameters to '%s'... " % fname, end="")
            parameters = (self.W, self.biases)
            pickle.dump(parameters, f)
            print("Done.")


    def load_parameters(self, fname=FNAME):
        """
        Loads the weights and biases from a file called `fname`.
        """
        with open(fname, "rb") as f:
            print("Loading parameters from '%s'... " % fname, end="")
            parameters = pickle.load(f)
            self.W, self.biases = parameters
            print("Done.")



