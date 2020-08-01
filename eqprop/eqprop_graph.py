
import torch


def create_ffnn_graph(layer_sizes):
    """
    Create a feed forward neural net graph.
    """
    # Initialize nodes and edges
    edges = []
    for out_layer, weights_size in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        out_size, in_size = weights_size
        for i in range(out_size + 1):  # plus bias
            for j in range(in_size):
                edge = ((out_layer,i), (out_layer+1,j))
                edges.append(edge)

    # Save input nodes, output nodes, and biases nodes
    input_nodes = set((0,i) for i in range(layer_sizes[0]))
    output_nodes = set((2,i) for i in range(layer_sizes[-1]))
    biases = set(enumerate(layer_sizes))

    return edges, input_nodes, output_nodes, biases


class EqPropGraph:
    def __init__(self, edges=[],
        input_nodes=set(), output_nodes=set(), biases=set()):
        """
        This version of eqprop is just for testing.
        It's very slow but it seems to work.
        It only works on a single batch (no batch dimension).
        It ignores hyperparameters and hardcode them instead.
        The code is a bit ugly and I'm not sorry.
        """
        self.states = {}
        self.W = {}
        self.biases = biases
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

        for out_node, in_node in edges:
            if out_node not in self.W:
                self.W[out_node] = {}
            # Close enough to Glorot-Bengio init
            self.W[out_node][in_node] = 0.05 * torch.randn(1).item()
            self.states[out_node] = 0
            self.states[in_node] = 0

        for node in self.biases:
            self.states[node] = 1


    def clamp_input(self, x):
        for i, node in enumerate(self.input_nodes):
            self.states[node] = x[i]


    def energy(self, states):
        nodes_energy = sum(s*s for s in states.values())
        edges_energy = sum(self.W[out_node][in_node] * states[out_node] * states[in_node]
                           for out_node in self.W for in_node in self.W[out_node])
        return edges_energy + 0.5 * nodes_energy


    def cost(self, states, y):
        cost = sum((states[node] - y[i])**2 for i, node in enumerate(self.output_nodes))
        return cost

    def output_state(self):
        return torch.tensor([v for _, v in self.output_nodes])


    def step(self, states, y=None):

        dt = 0.5
        beta = 0.5
        states_old = states.copy()

        for out_node in self.W:
            # Perturb states one edge at a time
            for in_node in self.W[out_node]:
                # No rho grad because we assume s is always in [0,1]
                states[in_node] += dt * self.W[out_node][in_node] * states[out_node]

        for node in states:
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

        for out_node in self.W:
            for in_node in self.W[out_node]:
                W_grad = clamped_states[out_node] * clamped_states[in_node] \
                    - free_states[out_node] * free_states[in_node]
                self.W[out_node][in_node] += lr / beta * W_grad


    def eqprop(self, x, y, train=True):

        x = [x[0, i] for i in range(x.size(-1))]
        y = [y[0, i] for i in range(y.size(-1))]

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



