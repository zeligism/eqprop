
import time
import torch

from training_configs import *
from dataset import dataset


class Net:
    def __init__(self, layers_sizes=LAYER_SIZES,
        learning_rates=LEARNING_RATES, beta=BETA, rho=RHO):
        """
        An equilibrium propagation network that works on PyTorch.
        """
        self.layers_sizes = layers_sizes
        self.learning_rates = learning_rates
        self.beta = beta
        self.rho = rho

        # Lists storing the weights, biases, and neuron-states of the network
        self.W = [
            torch.randn([BATCH_SIZE, l1, l2])
            for l1, l2 in zip(layers_sizes[:-1], layers_sizes[1:])]
        self.biases = [
            torch.zeros([BATCH_SIZE, l])
            for l in layers_sizes[:-1]]
        self.states = [
            torch.zeros([BATCH_SIZE, l], requires_grad=True)
            for l in layers_sizes]

    
    def energy(self, states):
        """
        Calculates the energy of the network.
        """
        rho = self.rho

        energy = 0

        for i in range(len(states)):
            # Sum of s_i * s_i for all i
            energy += 0.5 * torch.sum(states[i] * states[i], dim=-1)

        for i in range(len(states) - 1):
            # Sum of W_ij * rho(s_i) * rho(s_j) for all i, j
            wi_times_si = rho(states[i]).unsqueeze(dim=-2) @ self.W[i]
            energy -= torch.sum(wi_times_si.squeeze() * rho(states[i+1]), dim=-1)
            # Sum of bias_i * rho(s_i)
            energy -= torch.sum(self.biases[i] * rho(states[i]), dim=-1)

        return energy.mean()


    def cost(self, y):
        """
        Calculates the cost between the state of the last layer of the network
        with the output y. The cost is just the distance (L2 loss).
        """
        cost = torch.sum((self.output_state() - y) ** 2, dim=-1)
        
        return cost.mean()


    def clamp_input(self, x):
        """
        The following function simply clamps an input to the network.
        """
        self.states[0] = x


    def output_state(self):
        """
        Just returns the output of the network (last layer).
        """
        return self.states[-1]


    def zero_grad(self):
        """
        Zero out gradients of all parameters of the network.
        """
        [w.grad.data.zero_() for w in self.W      if w.grad is not None]
        [b.grad.data.zero_() for b in self.biases if b.grad is not None]
        [s.grad.data.zero_() for s in self.states if s.grad is not None]


    def step(self, y=None, dt=DELTA):
        """
        Make one step of duration dt.
        """

        # First zero the gradients
        self.zero_grad()

        # Calculate the total energy with the cost if y is given
        energy = self.energy(self.states)
        if y is not None:
            energy += self.beta * self.cost(y)

        # Calculate the gradients
        energy.backward()

        # Update states
        for i in range(1, len(self.states)):
            # Notice the negative sign because ds/dt = -dE/ds (partial d)
            self.states[i] = self.states[i] - dt * self.states[i].grad
            # We re-attach the states after updating them to reset gradient path
            self.states[i] = self.states[i].clamp(0,1).detach().requires_grad_()

        return energy


    def eqprop(self, x, y, validation=False):
        """
        Trains the network on one example (x,y) using equilibrium propagation.
        """

        # First clamp the input
        self.clamp_input(x)

        # Run free phase
        for i in range(N_ITER_1):
            energy = self.step()

        if validation:
            return energy.item(), self.cost(y).item()

        # Collect states
        free_states = [s.data for s in self.states]

        # Run weakly clamped phase
        for i in range(N_ITER_2):
            energy = self.step(y)

        #print("Updating weights...")
        self.update_weights(free_states)

        return energy.item(), self.cost(y).item()
    

    def update_weights(self, free_states, dt=DELTA):
        """
        TODO
        """
        lr = self.learning_rates

        [w.requires_grad_() for w in self.W]
        [b.requires_grad_() for b in self.biases]

        self.zero_grad()
        free_energy = self.energy(free_states)
        clamped_energy = self.energy(self.states)
        energy = 1 / self.beta * (clamped_energy - free_energy)
        energy.backward()

        # Update weights and biases (note the negative sign)
        for i in range(len(self.W)):
            self.W[i] = self.W[i] - lr[i] * self.W[i].grad
            self.W[i] = self.W[i].detach().requires_grad_()

        for i in range(len(self.biases)):
            self.biases[i] = self.biases[i] - lr[i] * self.biases[i].grad
            self.biases[i] = self.biases[i].detach().requires_grad_()

        [w.detach_() for w in self.W]
        [b.detach_() for b in self.biases]


##############################################


def index_to_onehot(index, num_indices):
    onehot = torch.zeros(*index.size(), num_indices)
    onehot.scatter_(-1, index.unsqueeze(dim=-1), 1)

    return onehot


def train(net, trainloader):
    start_time = time.time()
    for epoch in range(EPOCHS):
        running_energy = running_cost = 0.0
        for i, data in enumerate(trainloader):
            x, y = data

            # Prepare data
            x = x.view(-1, LAYER_SIZES[0])
            y_onehot = index_to_onehot(y, LAYER_SIZES[-1])

            # Train on (x,y) using equilibrium propagation
            energy, cost = net.eqprop(x, y_onehot)

            running_energy += energy
            running_cost += cost
            if (i + 1) % CHECKPOINT == 0:
                avg_cost = running_cost / CHECKPOINT
                print("[%d, %d] energy = %.3f, cost = %.3f" % (epoch+1, i+1,
                    running_energy / CHECKPOINT, running_cost / CHECKPOINT))
                print("Time elapsed = %ds" % (time.time() - start_time))
                running_energy = running_cost = 0.0

    print("\nEPOCH: [%d/%d] energy = %.3f, cost = %.3f\n" % (epoch+1, EPOCHS, energy, cost))
            

def test(net, testloader):
    running_hits = iterations = 0
    for i, data in enumerate(testloader):
        x, y = data

        # Prepare data
        x = x.view(-1, LAYER_SIZES[0])
        y_onehot = index_to_onehot(y, LAYER_SIZES[-1])

        # Train on (x,y) using equilibrium propagation
        energy, cost = net.eqprop(x, y_onehot, validation=True)
        print("[%d] energy = %.3f, cost = %.3f" % (i, energy, cost))

        # Calculate hits
        _, index = net.output_state().max(dim=-1)
        hits = (y == index).sum().item()
        print("hits =", hits, "out of", BATCH_SIZE)

        running_hits += hits
        iterations += 1

    print("\nAverage hits =", running_hits / iterations)
    print("Error =", 100 * (1 - running_hits / (iterations * BATCH_SIZE)))


def main():
    # Set random seed if given
    torch.manual_seed(RANDOM_SEED or torch.initial_seed())

    # Dataset
    trainloader, testloader = dataset(BATCH_SIZE)

    # Network
    eqprop_net = Net()

    # Train
    train(eqprop_net, trainloader)

    # Validate
    test(eqprop_net, testloader)


if __name__ == '__main__':
    main()







        