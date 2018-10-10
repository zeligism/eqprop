
import time
import torch

from training_configs import *
from dataset import dataset
from eqprop_torch import EqPropNet


def index_to_onehot(index, num_indices):
    onehot = torch.zeros(*index.size(), num_indices)
    onehot.scatter_(-1, index.unsqueeze(dim=-1), 1)

    return onehot


def train(net, trainloader):
    start_time = time.time()
    for epoch in range(EPOCHS):
        return
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
    eqprop_net = EqPropNet()

    # Train
    train(eqprop_net, trainloader)

    # Validate
    test(eqprop_net, testloader)


if __name__ == '__main__':
    main()







        