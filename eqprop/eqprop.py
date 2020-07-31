
import os
import pickle
import torch


class EqPropNet:
    def __init__(self, batch_size, layer_sizes, learning_rates,
        free_iters, clamped_iters, rho=lambda x: x.clamp(0,1), beta=1, dt=0.5):
        """
        An equilibrium propagation network that works on PyTorch.
        """

        # Training-specific hyperparameters
        self.lr = learning_rates
        self.free_iters = free_iters
        self.clamped_iters = clamped_iters
        # Dynamics-specific hyperparameters
        self.rho = rho
        self.beta = beta
        self.dt = dt

        # Initialize weights using Glorot-Bengio initialization
        self.weights = [torch.randn(l1, l2) * torch.tensor(2 / (l1 + l2)).sqrt()
            for l1, l2 in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [torch.zeros(l) for l in layer_sizes[1:]]

        # Initialize states to 0 (or I guess anything between 0 and 1 is ok)
        self.states = [torch.zeros(batch_size, l) for l in layer_sizes]

    
    def energy(self, states):
        """
        Calculates the energy of the network.
        """
        energy = 0
        for i in range(len(states)):
            # Sum of s_i * s_i for all i
            energy += 0.5 * torch.sum(states[i] * states[i], dim=-1)

        for i in range(len(self.weights)):
            # Sum of W_ij * rho(s_i) * rho(s_j) for all i, j
            energy -= torch.sum(
                (self.rho(states[i]) @ self.weights[i]) * self.rho(states[i+1]), dim=-1)
            # Sum of bias_i * rho(s_i)
            energy -= torch.sum(self.biases[i] * self.rho(states[i+1]), dim=-1)

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


    def step(self, states, y=None):
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
            # Notice the negative sign because ds/dt = -dE/ds
            states[i] = states[i] - self.dt * states[i].grad
            states[i].clamp_(0,1).detach_()

        return energy


    def update_weights(self, free_states, clamped_states):
        """
        TODO: doc
        """

        [w.requires_grad_() for w in self.weights]
        [b.requires_grad_() for b in self.biases]

        free_energy = self.energy(free_states).mean()
        clamped_energy = self.energy(clamped_states).mean()
        energy = 1 / self.beta * (clamped_energy - free_energy)
        energy.backward()

        # Update weights and biases (note the negative sign)
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - self.lr[i] * self.weights[i].grad
            self.weights[i].detach_()

        for i in range(len(self.biases)):
            self.biases[i] = self.biases[i] - self.lr[i] * self.biases[i].grad
            self.biases[i].detach_()


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



