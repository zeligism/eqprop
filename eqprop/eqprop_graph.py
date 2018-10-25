
import pickle
import torch


class EqPropGraph:
    def __init__(self, edges=[],
        input_nodes=set(), output_nodes=set(), biases=set()):
        """
        This version of eqprop is just for... uh, fun.
        It's very slow but it seems to work.
        It only works on a single batch (no batch dimension).
        It ignores hyperparameters and hardcode them instead.
        The code here is not written with maintenance in mind,
        and it lacks documentations. 
        """
        self.states = {}
        self.W = {}
        self.biases = biases
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

        for node1, node2 in edges:
            if node1 not in self.W:
                self.W[node1] = {}
            # Close enough to Glorot-Bengio init
            self.W[node1][node2] = 0.05 * torch.randn(1).item()
            self.states[node1] = 0
            self.states[node2] = 0

        for node in self.biases:
            self.states[node] = 1


    def clamp_input(self, x):
        for i, node in enumerate(self.input_nodes):
            self.states[node] = x[i]


    def energy(self, states):
        energy = 0

        for s in states.values():
            energy += s*s
        energy *= 0.5

        for node1 in self.W.keys():
            for node2 in self.W[node1].keys():
                energy -= self.W[node1][node2] * states[node1] * states[node2]

        return energy


    def cost(self, states, y):
        cost = 0
        for i, node in enumerate(self.output_nodes):
            diff = states[node] - y[i]
            cost += diff * diff

        return cost


    def step(self, states, y=None):

        dt = 0.5
        beta = 0.5
        states_old = states.copy()

        for node1 in self.W.keys():
            # Perturb states one edge at a time
            for node2 in self.W[node1].keys():
                # No rho grad because we assume s is always in [0,1]
                states[node2] += dt * self.W[node1][node2] * states[node1]

        for node in states.keys():
            if node not in self.input_nodes and node not in self.biases:
                states[node] -= dt * states_old[node]
                if states[node] < 0 or states[node] > 1:
                    states[node] = 0

        # Update last layer in weakly clamped phase
        if y is not None:
            for i, node in enumerate(self.output_nodes):
                # TODO y has a structure?
                states[node] += dt * (y[i] - states[node])



    def update_weights(self, free_states, clamped_states):

        lr = 0.05
        beta = 0.5

        for node1 in self.W.keys():
            for node2 in self.W[node1].keys():
                W_grad = clamped_states[node1] * clamped_states[node2] \
                    - free_states[node1] * free_states[node2]
                self.W[node1][node2] += lr / beta * W_grad


    def eqprop(self, x, y, train=True):

        # First clamp the input
        self.clamp_input(x)

        # Run free phase
        for i in range(20):
            self.step(self.states)

        if train:
            # Collect states and perturb them to the weakly clamped y
            clamped_states = self.states.copy()
            for i in range(4):
                self.step(clamped_states, y)

            # Update weights
            self.update_weights(self.states, clamped_states)

        return self.energy(self.states), self.cost(self.states, y)



